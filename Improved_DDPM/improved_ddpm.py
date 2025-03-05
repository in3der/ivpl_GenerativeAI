# References:
# https://nn.labml.ai/diffusion/ddpm/index.html
# https://huggingface.co/blog/annotated-diffusion

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm
import contextlib
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

def plt_to_pil(fig):
    """Matplotlib figure를 PIL 이미지로 변환"""
    fig.canvas.draw()
    img_array = np.array(fig.canvas.renderer.buffer_rgba())
    return Image.fromarray(img_array)

def save_image(image, path):
    """PIL 이미지를 저장"""
    image.save(path)

def get_linear_beta_schedule(init_beta, fin_beta, n_diffusion_steps):
    """선형 beta 스케줄 계산"""
    return torch.linspace(init_beta, fin_beta, n_diffusion_steps)


class ImprovedDDPM(nn.Module):

    def get_linear_beta_schedule(init_beta, fin_beta, n_diffusion_steps=1000):
        """선형 beta 스케줄 계산"""
        return torch.linspace(init_beta, fin_beta, n_diffusion_steps)
    def get_cos_beta_schedule(self, s=0.008):
        """코사인 스케줄을 사용한 beta 값 계산"""
        steps = self.n_diffusion_steps
        diffusion_step = torch.linspace(0, steps - 1, steps, device=self.device)
        alphas = torch.cos(((diffusion_step / steps) + s) / (1 + s) * torch.pi / 2) ** 2
        alphas = alphas / alphas[0]  # 정규화
        self.alpha_bar = alphas
        # alpha_bar에서 beta 계산
        self.prev_alpha_bar = torch.cat([torch.ones(1, device=self.device), self.alpha_bar[:-1]])
        self.beta = 1 - (self.alpha_bar / self.prev_alpha_bar)
        self.beta = torch.clip(self.beta, 0, 0.999)

        # beta_tilde 계산
        self.beta_tilde = ((1 - self.prev_alpha_bar) / (1 - self.alpha_bar)) * self.beta

    def __init__(
            self,
            model,
            img_size,
            device,
            n_subsequence_steps=100,
            image_channels=3,
            n_diffusion_steps=1000,
            vlb_weight=0.001,
    ):
        super().__init__()

        self.img_size = img_size
        self.device = device
        self.image_channels = image_channels
        self.n_subsequence_steps = n_subsequence_steps
        self.n_diffusion_steps = n_diffusion_steps
        self.vlb_weight = vlb_weight

        self.model = model.to(device)

        # CUDA 초기화 후 스케줄 계산
        with torch.cuda.device(device):
            self.get_cos_beta_schedule()

        # 서브시퀀스 스텝 계산을 디바이스에 맞게 수정
        self.subsequence_step = torch.linspace(
            0, self.n_diffusion_steps - 1, self.n_subsequence_steps,
            dtype=torch.long, device=self.device
        )

    @staticmethod
    def index(x, diffusion_step):
        return x[torch.clip(diffusion_step, min=0)][:, None, None, None]

    def sample_noise(self, batch_size):
        return torch.randn(
            size=(batch_size, self.image_channels, self.img_size, self.img_size),
            device=self.device,
        )

    def sample_diffusion_step(self, batch_size):
        return torch.randint(
            0, self.n_diffusion_steps, size=(batch_size,), device=self.device,
        )

    def batchify_diffusion_steps(self, diffusion_step_idx, batch_size):
        return torch.full(
            size=(batch_size,),
            fill_value=diffusion_step_idx,
            dtype=torch.long,
            device=self.device,
        )

    def get_model_var(self, v, diffusion_step):
        return self.index(
            v * torch.log(self.beta) + (1 - v) * torch.log(self.beta_tilde),
            diffusion_step=diffusion_step,
        )

    def forward(self, noisy_image, diffusion_step):
        """forward 함수 개선"""
        # diffusion_step의 shape 확인 및 수정
        if diffusion_step.dim() == 1:
            diffusion_step = diffusion_step.view(-1, 1, 1, 1)
        return self.model(noisy_image=noisy_image, diffusion_step=diffusion_step)

    def get_mu_tilde(self, ori_image, noisy_image, diffusion_step):
        return self.index(
            (self.prev_alpha_bar ** 0.5) / (1 - self.alpha_bar),
            diffusion_step=diffusion_step,
        ) * ori_image + self.index(
            ((self.alpha_bar ** 0.5) * (1 - self.prev_alpha_bar)) / (1 - self.alpha_bar),
            diffusion_step=diffusion_step,
        ) * noisy_image

    @torch.inference_mode()
    def sample(self, batch_size):
        """샘플링 과정 개선"""
        with torch.no_grad():
            x = self.sample_noise(batch_size=batch_size)
            for subsequence_idx in tqdm(range(self.n_subsequence_steps - 1, -1, -1)):
                batched_subsequence_idx = self.batchify_diffusion_steps(
                    diffusion_step_idx=subsequence_idx,
                    batch_size=batch_size
                )

                # 현재 스텝과 이전 스텝 인덱스 계산
                cur_step = self.subsequence_step[batched_subsequence_idx]
                prev_step = self.subsequence_step[torch.max(batched_subsequence_idx - 1, torch.tensor(0))]

                # alpha_bar 값들 가져오기
                alpha_bar_t = self.alpha_bar[cur_step]
                prev_alpha_bar_t = self.alpha_bar[prev_step]
                beta_t = 1 - alpha_bar_t / prev_alpha_bar_t

                # 노이즈 예측 및 평균/분산 계산
                pred_noise = self(noisy_image=x, diffusion_step=cur_step)

                # 수치적 안정성을 위한 epsilon 추가
                eps = 1e-8
                model_mean = (1 / torch.sqrt(1 - beta_t + eps)) * (
                        x - (beta_t / torch.sqrt(1 - alpha_bar_t + eps)) * pred_noise
                )

                if subsequence_idx > 0:  # 마지막 스텝이 아닌 경우에만 노이즈 추가
                    noise = self.sample_noise(batch_size=batch_size)
                    model_var = beta_t
                    x = model_mean + torch.sqrt(model_var) * noise
                else:
                    x = model_mean

        return x


if __name__ == "__main__":
    n_diffusion_steps = 1000
    init_beta = 0.0001
    fin_beta = 0.02
    linear_beta = get_linear_beta_schedule(
        init_beta=init_beta, fin_beta=fin_beta, n_diffusion_steps=n_diffusion_steps,
    )
    cos_beta = ImprovedDDPM.get_cos_beta_schedule()

    linear_alpha = 1 - linear_beta
    linear_alpha_bar = torch.cumprod(linear_alpha, dim=0)

    cos_alpha = 1 - cos_beta
    cos_alpha_bar = torch.cumprod(cos_alpha, dim=0)
    # linear_alpha_bar[0]
    # cos_alpha_bar[0]

    fig, axes = plt.subplots(1, 1, figsize=(5, 3))
    line2 = axes.plot(linear_alpha_bar.numpy(), label="Linear")
    line2 = axes.plot(cos_alpha_bar.numpy(), label="Cosine")
    # line2 = axes.plot((linear_alpha_bar.numpy() ** 0.5))
    # line2 = axes.plot((cos_alpha_bar.numpy() ** 0.5))
    axes.legend(fontsize=6)
    axes.tick_params(labelsize=7)
    fig.tight_layout()
    image = plt_to_pil(fig)
    # image.show()
    save_image(image, path="/home/ivpl-d29/myProject/GenerativeAI/Improved_DDPM/beta_schedules.jpg")