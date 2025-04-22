import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


# Zero-initialize 모든 파라미터
def make_zero(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

class ControlNetLike(nn.Module):

    def __init__(self):
        super().__init__()
        dims = 2
        hint_channels=3
        model_channels=3
        self.F = nn.Sequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            make_zero(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        # self.zero1 = nn.Sequential(
        #     make_zero(nn.Conv2d(3, 3, 1)),  # 입력/출력 채널 3으로 수정
        #     nn.ReLU()
        # )
        self.zero1 = make_zero(nn.Conv2d(3, 3, 1))
        self.relu = nn.ReLU()
        self.zero2 = make_zero(nn.Conv2d(3, 3, 1, padding=0))  # f_out이 1채널이므로 여기에 맞춤

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(48, 1)



    def forward(self, x, cond):
        z1 = self.zero1(x)
        f_out = self.F(x+z1)
        z2 = self.zero2(f_out)
        out = self.F(x) + z2  # ControlNet style output
        out = self.flatten(out)
        out = self.fc(out)
        return out

# 데이터 준비
x = torch.randn(3, 3, 28, 28)
cond = torch.randn(3, 3, 28, 28)
target = torch.tensor([[1.0], [1.0], [1.0]])  # shape = (3, 1)

# 모델과 optimizer
model = ControlNetLike()
optimizer = optim.SGD(model.parameters(), lr=0.1)

import matplotlib.pyplot as plt

# # 모델구조 시각화
# from torchviz import make_dot
#
# # 예시 입력
# x = torch.randn(1, 3, 28, 28)
# cond = torch.randn(1, 3, 28, 28)
#
# model = ControlNetLike()
#
# # 그래프 생성
# output = model(x, cond)
# dot = make_dot(output, params=dict(model.named_parameters()))
# dot.format = 'png'
# dot.directory = './'  # 현재 디렉토리에 저장
# dot.render("WhyZeroConvWeightTrained_model", cleanup=True)
#
# print("✅ 모델 구조가 'WhyZeroConvWeightTrained_model.png'로 저장되었습니다!")


# 그래프 저장용 리스트
losses = []
w1_vals = []
b1_vals = []
w2_vals = []
b2_vals = []
g1_vals = []
g2_vals = []

print("===> Training Started (ControlNet style)...")
for step in range(10):
    optimizer.zero_grad()
    output = model(x, cond)
    loss = F.mse_loss(output, target)
    loss.backward()
    optimizer.step()

    # weight/gradient 저장
    w1 = model.zero1.weight.data[0, 0, 0, 0].item()
    b1 = model.zero1.bias.data[0].item()
    g1 = model.zero1.weight.grad[0, 0, 0, 0].item()

    w2 = model.zero2.weight.data[0, 0, 0, 0].item()
    b2 = model.zero2.bias.data[0].item()
    g2 = model.zero2.weight.grad[0, 0, 0, 0].item()

    # 로그 출력
    print(f"Step {step + 1:2d}: loss = {loss:.6f}, w1 = {w1:.20f}, b1 = {b1:.20f}, w2 = {w2:.6f}, b2 = {b2:.6f}")
    print("g1:", g1, "g2:", g2)

    # 저장
    losses.append(loss.item())
    w1_vals.append(w1)
    b1_vals.append(b1)
    w2_vals.append(w2)
    b2_vals.append(b2)
    g1_vals.append(g1)
    g2_vals.append(g2)

# ✅ 시각화
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

axs[0, 0].plot(losses, label="Loss")
axs[0, 0].set_title("Loss over Steps")
axs[0, 0].set_xlabel("Step")
axs[0, 0].set_ylabel("Loss")
axs[0, 0].legend()

axs[0, 1].plot(w1_vals, label="w1 (zero1)", color="red")
axs[0, 1].plot(w2_vals, label="w2 (zero2)", color="blue")
axs[0, 1].set_title("Weight Values")
axs[0, 1].set_xlabel("Step")
axs[0, 1].set_ylabel("Weight")
axs[0, 1].legend()

axs[1, 0].plot(g1_vals, label="grad w1 (zero1)", color="red")
axs[1, 0].plot(g2_vals, label="grad w2 (zero2)", color="blue")
axs[1, 0].set_title("Gradients")
axs[1, 0].set_xlabel("Step")
axs[1, 0].set_ylabel("Gradient")
axs[1, 0].legend()

axs[1, 1].plot(b1_vals, label="b1", color="red")
axs[1, 1].plot(b2_vals, label="b2", color="blue")
axs[1, 1].set_title("Bias Values")
axs[1, 1].set_xlabel("Step")
axs[1, 1].set_ylabel("Bias")
axs[1, 1].legend()

plt.tight_layout()
#plt.show()
plt.savefig("WhyZeroConvWeightTrained.png")




