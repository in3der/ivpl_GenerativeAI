import torch
import open_clip
from dalle2_pytorch import DALLE2, DiffusionPriorNetwork, DiffusionPrior, Unet, Decoder, CLIP
from torchvision.utils import save_image

# open_clip 모델 로드
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32',
    pretrained='laion2b_s34b_b79k'
)
model = model.cuda()
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# CLIP 모델 초기화 (open_clip의 모델 구조에 맞게 조정)
clip = CLIP(
    dim_text = 512,
    dim_image = 512,
    dim_latent = 512,
    num_text_tokens = 49408,
    text_enc_depth = 6,
    text_seq_len = 256,
    text_heads = 8,
    visual_enc_depth = 6,
    visual_image_size = 256,
    visual_patch_size = 32,
    visual_heads = 8
).cuda()

# open_clip 모델의 가중치를 dalle2-pytorch의 CLIP 모델에 로드
# 이 부분은 모델 구조에 따라 다를 수 있으며 수동 매핑이 필요할 수 있음
state_dict = model.state_dict()
clip_state_dict = clip.state_dict()

# 가중치 매핑 로직 (대략적인 예시, 실제로는 더 복잡할 수 있음)
for name, weights in state_dict.items():
    if name in clip_state_dict:
        clip_state_dict[name] = weights

clip.load_state_dict(clip_state_dict)

# 토큰화 함수 정의
def tokenize_text(text):
    return tokenizer(text).cuda()

# 나머지 코드는 동일
prior_network = DiffusionPriorNetwork(
    dim = 512,
    depth = 6,
    dim_head = 64,
    heads = 8
).cuda()

diffusion_prior = DiffusionPrior(
    net = prior_network,
    clip = clip,
    timesteps = 10000,
    sample_timesteps = 64,
    cond_drop_prob = 0.2
).cuda()

# (이하 코드 동일)
unet1 = Unet(
    dim = 128,
    image_embed_dim = 512,
    text_embed_dim = 512,
    cond_dim = 128,
    channels = 3,
    dim_mults=(1, 2, 4, 8),
    cond_on_text_encodings = True
).cuda()

unet2 = Unet(
    dim = 16,
    image_embed_dim = 512,
    cond_dim = 128,
    channels = 3,
    dim_mults = (1, 2, 4, 8, 16)
).cuda()

decoder = Decoder(
    unet = (unet1, unet2),
    image_sizes = (128, 256),
    clip = clip,
    timesteps = 1000,
    image_cond_drop_prob = 0.1,
    text_cond_drop_prob = 0.5
).cuda()

# DALL-E 2 모델 초기화
dalle2 = DALLE2(
    prior = diffusion_prior,
    decoder = decoder
)

# 이미지 생성
prompt = "Cute puppy chasing a squirrel"
images = dalle2(
    [prompt],
    cond_scale = 2.  # 조건부 가이던스 강도
)

# 이미지 후처리 및 저장
images = (images + 1) / 2.0  # [-1, 1] 범위를 [0, 1]로 정규화
images = images.clamp(0, 1)  # 값을 0과 1 사이로 고정
save_image(images, 'generated_image22.png')
print("이미지 생성 완료!")