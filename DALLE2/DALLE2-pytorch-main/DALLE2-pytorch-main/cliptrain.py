import torch
from torch.cuda.amp import autocast, GradScaler
from dalle2_pytorch import Unet, Decoder, CLIP

# GPU 메모리 최적화 설정
torch.backends.cudnn.benchmark = True

# 하프 프리시전 및 그래디언트 스케일러 초기화
scaler = GradScaler()

# CLIP 모델 최적화
clip = CLIP(
    dim_text=256,  # 차원 축소
    dim_image=256,  # 차원 축소
    dim_latent=256,  # 차원 축소
    num_text_tokens=49408,
    text_enc_depth=1,  # 레이어 축소
    text_seq_len=256,
    text_heads=8,
    visual_enc_depth=1,  # 레이어 축소
    visual_image_size=256,
    visual_patch_size=32,
    visual_heads=8
).cuda()  # float16으로 변환

# Unet 모델 최적화
unet = Unet(
    dim=64,  # 차원 축소
    image_embed_dim=256,
    cond_dim=64,  # 차원 축소
    channels=3,
    dim_mults=(1, 2, 4, 8)
).cuda()  # float16으로 변환

# Decoder 모델 최적화
decoder = Decoder(
    unet=unet,
    clip=clip,
    timesteps=100,
    image_cond_drop_prob=0.1,
    text_cond_drop_prob=0.5
).cuda()  # float16으로 변환

# 옵티마이저 설정
optimizer = torch.optim.Adam(
    list(clip.parameters()) +
    list(unet.parameters()) +
    list(decoder.parameters()),
    lr=1e-4
)

# 학습 하이퍼파라미터
batch_size = 2  # 배치 크기 축소
accumulation_steps = 4  # 그래디언트 누적 스텝
total_steps = 1000  # 총 학습 스텝

# 학습 루프
for step in range(total_steps):
    # 입력 데이터 생성 (배치 크기 축소 및 float16)
    images = torch.randn(batch_size, 3, 256, 256).half().cuda()

    # 혼합 정밀도 학습
    with autocast():
        # 디코더에 이미지 입력
        loss = decoder(images)

        # 그래디언트 스케일링 및 누적
        scaled_loss = scaler.scale(loss / accumulation_steps)
        scaled_loss.backward()

        # 주기적으로 옵티마이저 업데이트
        if (step + 1) % accumulation_steps == 0:
            # 그래디언트 스케일 조정 및 최적화
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    # 진행 상황 출력
    if (step + 1) % 100 == 0:
        print(f"Step {step + 1}/{total_steps}, Loss: {loss.item():.4f}")

# 모델 저장
torch.save({
    'clip_state_dict': clip.state_dict(),
    'unet_state_dict': unet.state_dict(),
    'decoder_state_dict': decoder.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}, 'model_checkpoint.pth')

print("학습 완료!")