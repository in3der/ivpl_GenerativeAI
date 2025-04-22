import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
from PIL import Image
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------
# 1. 모델 불러오기 (ControlNet + Stable Diffusion)
# ---------------------------------------------
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
).cuda()

for name, module in controlnet.named_modules():
    if "zero_convs" in name:
        print(f"ZeroConv layer: {name}")
        for param_name, param in module.named_parameters():
            print(f"  {param_name}: mean={param.data.mean().item()}, std={param.data.std().item()}")


pipe.safety_checker = None  # NSFW 필터링 제거 (옵션)

# ---------------------------------------------
# 2. Hook 등록: zero conv의 출력 관측
# ---------------------------------------------
hook_outputs = {}

def register_hooks(module, name_prefix=""):
    for name, layer in module.named_children():
        full_name = f"{name_prefix}.{name}" if name_prefix else name
        #print("full name: ", full_name, layer)
        if isinstance(layer, torch.nn.Conv2d) :
            def hook_fn(m, i, o, full_name=full_name):
                hook_outputs[full_name] = o.detach().cpu()
                print(f"[HOOK] {full_name} output → mean: {o.mean().item():.6f}, std: {o.std().item():.6f}")
            layer.register_forward_hook(hook_fn)
        register_hooks(layer, full_name)

print("🔍 Registering hooks for zero conv layers...")
register_hooks(controlnet)


# ---------------------------------------------
# 3. 입력 이미지 & 조건 이미지 준비
# ---------------------------------------------
# 예시: 임의의 흑백 입력 이미지 생성 (원한다면 실제 이미지 사용)
image_path = "/home/ivpl-d29/dataset/imagenet_test/whippet.jpg"

# 이미지 로드 및 전처리
image = Image.open(image_path).convert("RGB")
transform = T.Compose([
    T.Resize((512, 512)),             # 모델에 맞는 사이즈로 조정
    T.ToTensor(),                     # [0,1] 범위로 tensor 변환
    T.Normalize([0.5]*3, [0.5]*3),    # [-1,1] 범위로 normalize (ControlNet 용)
])
image_tensor = transform(image).unsqueeze(0).half().cuda()  # (1, 3, 512, 512)

dummy_image = image_tensor
dummy_condition = dummy_image.clone()  # 동일한 걸 조건으로 사용

# 조건 이미지를 ControlNet에 맞게 resize
from torchvision.transforms import Resize
dummy_condition = Resize((512, 512))(dummy_condition)

# ---------------------------------------------
# 4. Forward pass 실행
# ---------------------------------------------
with torch.no_grad():
    _ = pipe(
        prompt="a futuristic city",
        image=dummy_condition,
        num_inference_steps=5  # 빠르게 보기 위해 1 step
    )

# ---------------------------------------------
# 5. 결과 시각화
# ---------------------------------------------
# ZeroConv output 중 하나를 시각화
print("visualization")
for name, tensor in hook_outputs.items():
    print("name : ", name)
    #print("tensor : ", tensor)
    #print("hoot outputs : ", hook_outputs)
    print(f"\n🔍 Visualizing output from: {name} — shape: {tensor.shape}")
    if tensor.dim() == 4:
        # 첫 번째 채널만 시각화
        vis = tensor[0, 0].numpy()
        plt.imshow(vis, cmap='viridis')
        plt.title(name)
        plt.colorbar()
        plt.savefig("outputs_" + name + ".png")
        print("saved")
        break  # 하나만 시각화, 여러 개 보려면 break 제거

# ---------------------------------------------
# 6. ZeroConv weight 관측
# 그런데 이미 다 학습된 모듈들을 불러오는거라 다 mean, std 값 있음 주의
# ---------------------------------------------
print("\n📦 Zero conv layer weights (initial snapshot):")
for name, param in controlnet.named_parameters():
    #print(name, param.shape)
    if 'conv_in' in name and 'weight' in name:
        print(f"{name} → mean: {param.data.mean().item():.6f}, std: {param.data.std().item():.6f}")
    if 'proj_in' in name and 'weight' in name:
        print(f"{name} → mean: {param.data.mean().item():.6f}, std: {param.data.std().item():.6f}")
    if 'proj_out' in name and 'weight' in name:
        print(f"{name} → mean: {param.data.mean().item():.6f}, std: {param.data.std().item():.6f}")
