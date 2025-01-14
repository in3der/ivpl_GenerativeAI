from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

# ----------------------------
# Argument Parser 설정
# ----------------------------
parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-mps', action='store_true', default=False,
                    help='disables macOS GPU training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

# CUDA 사용 여부를 결정 (args.no_cuda가 False이고 CUDA가 사용 가능할 경우 사용)
args.cuda = not args.no_cuda and torch.cuda.is_available()
# macOS GPU(MPS) 사용 여부 결정
use_mps = not args.no_mps and torch.backends.mps.is_available()

# ----------------------------
# 랜덤 시드 설정
# ----------------------------
torch.manual_seed(args.seed)

# ----------------------------
# 디바이스 설정 (CUDA, MPS 또는 CPU)
# ----------------------------
if args.cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# ----------------------------
# DataLoader 설정
# ----------------------------
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# MNIST 데이터셋 로드 (훈련용)
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

# MNIST 데이터셋 로드 (테스트용)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=False, **kwargs)


# ----------------------------
# VAE 모델 정의
# ----------------------------
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # 인코더 부분
        self.fc1 = nn.Linear(784, 400)  # 입력: 28x28 이미지를 펼쳐서 784 크기의 벡터로 변환
        self.fc21 = nn.Linear(400, 20)  # 평균 벡터 \( \mu \)
        self.fc22 = nn.Linear(400, 20)  # 분산 벡터 \( \log\sigma^2 \)

        # 디코더 부분
        self.fc3 = nn.Linear(20, 400)  # 잠재 변수 \( z \)를 400차원으로 복원
        self.fc4 = nn.Linear(400, 784)  # 400차원을 다시 784 크기로 복원

    def encode(self, x):
        # 입력 데이터를 은닉 공간으로 매핑
        h1 = F.relu(self.fc1(x))  # 첫 번째 은닉층
        return self.fc21(h1), self.fc22(h1)  # 평균과 분산 반환

    def reparameterize(self, mu, logvar):
        # reparameterize trick 사용하여 x 샘플링
        std = torch.exp(0.5 * logvar)  # 표준 편차
        eps = torch.randn_like(std)  # 표준 정규분포에서 샘플링
        return mu + eps * std  # \( z = \mu + \sigma \cdot \epsilon \)

    def decode(self, z):
        # 잠재 변수 \( z \)로부터 데이터를 복원
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))  # 복원된 데이터 출력 (0~1 사이 값)

    def forward(self, x):
        # 전체 순전파 과정
        mu, logvar = self.encode(x.view(-1, 784))  # 인코딩
        z = self.reparameterize(mu, logvar)  # 재매개변수화
        return self.decode(z), mu, logvar  # 복원된 데이터와 매개변수 반환


# ----------------------------
# 모델 및 옵티마이저 설정
# ----------------------------
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# ----------------------------
# 손실 함수 정의
# ----------------------------
def loss_function(recon_x, x, mu, logvar):
    # reconstruction error (Binary Cross Entropy)
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # KL Divergence : latent distribution 정규화
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # 두 손실의 합 반환
    return BCE + KLD


# ----------------------------
# 학습 루프 정의
# ----------------------------
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)  # 순전파
        loss = loss_function(recon_batch, data, mu, logvar)  # 손실 계산
        loss.backward()  # 역전파
        train_loss += loss.item()
        optimizer.step()  # 가중치 업데이트

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


# ----------------------------
# 테스트 루프 정의
# ----------------------------
def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                # 첫 번째 배치에서 복원된 이미지 저장
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                           'results/reconstruction_' + str(epoch) + '.png', nrow=n)
## reconstruction : 입력 데이터를 모델이 인코딩(encoding)하고, 다시 디코딩(decoding)하여 '복원'한 결과

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


# ----------------------------
# 메인 실행 루프
# ----------------------------
if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)  # 훈련
        test(epoch)  # 테스트
        with torch.no_grad():
            # 잠재 공간에서 샘플링하여 새 이미지를 생성
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')
# sample : 잠재 공간(latent space)에서 무작위로 생성한 z를 디코더(decoder)에 전달하여 생성된 이미지(새롭게 '생성'된 이미지)