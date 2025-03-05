import torch
import argparse
import os

from utils import get_device, image_to_grid, save_image
from unet import UNet
from improved_ddpm import ImprovedDDPM

# 환경 변수 설정
BASE_DIR = "/home/ivpl-d29/myProject/GenerativeAI/Improved_DDPM"
MODEL_PARAMS = os.path.join(BASE_DIR, "64x64_diffusion.pt")
SAVE_DIR = os.path.join(BASE_DIR, "samples")
IMG_SIZE = 64
BATCH_SIZE = 2


def get_args():
    return argparse.Namespace(
        MODE="normal",
        MODEL_PARAMS=MODEL_PARAMS,
        SAVE_PATH=os.path.join(SAVE_DIR, "0.jpg"),
        IMG_SIZE=IMG_SIZE,
        BATCH_SIZE=BATCH_SIZE,
    )


def main():
    # CUDA 메모리 관리 최적화 설정
    torch.backends.cudnn.benchmark = True
    torch.set_printoptions(linewidth=70)

    # 다른 CUDA 작업 전에 CUDA 초기화
    DEVICE = get_device()
    torch.cuda.init()  # CUDA 명시적 초기화
    print(f"[ DEVICE: {DEVICE} ]")

    # CUDA 캐시 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    args = get_args()

    # 명시적 디바이스 배치로 모델 초기화
    net = UNet()
    net = net.to(DEVICE)  # ImprovedDDPM 생성 전에 네트워크를 디바이스로 이동

    model = ImprovedDDPM(model=net, img_size=args.IMG_SIZE, device=DEVICE)

    # 안전한 모델 가중치 로딩
    state_dict = torch.load(
        str(args.MODEL_PARAMS),
        map_location=DEVICE,
        weights_only=True  # FutureWarning 해결
    )
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()  # 평가 모드로 설정

    try:
        if args.MODE == "denoising_process":
            model.vis_denoising_process(
                batch_size=args.BATCH_SIZE,
                save_path=args.SAVE_PATH,
            )
        else:
            if args.MODE == "normal":
                with torch.no_grad():  # 불필요한 그래디언트 계산 방지
                    gen_image = model.sample(args.BATCH_SIZE)
                    gen_grid = image_to_grid(gen_image, n_cols=max(1, int(args.BATCH_SIZE ** 0.5)))
                    gen_grid.show()
            elif args.MODE == "interpolation":
                gen_image = model.interpolate(
                    data_dir=args.DATA_DIR,
                    image_idx1=args.IMAGE_IDX1,
                    image_idx2=args.IMAGE_IDX2,
                    interpolate_at=args.INTERPOLATE_AT,
                    n_points=args.N_POINTS,
                )
                gen_grid = image_to_grid(gen_image, n_cols=args.N_POINTS + 2)
                save_image(gen_grid, save_path=args.SAVE_PATH)
            elif args.MODE == "coarse_to_fine":
                gen_image = model.coarse_to_fine_interpolate(
                    data_dir=args.DATA_DIR,
                    image_idx1=args.IMAGE_IDX1,
                    image_idx2=args.IMAGE_IDX2,
                    n_rows=args.N_ROWS,
                    n_points=args.N_POINTS,
                )
                gen_grid = image_to_grid(gen_image, n_cols=args.N_POINTS + 2)
                save_image(gen_grid, save_path=args.SAVE_PATH)
    except RuntimeError as e:
        print(f"CUDA 오류 발생: {str(e)}")
        torch.cuda.empty_cache()  # CUDA 메모리 정리
        raise  # 추가 컨텍스트와 함께 예외 다시 발생


if __name__ == "__main__":
    main()