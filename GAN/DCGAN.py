import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

# Set random seed for reproducibility
torch.manual_seed(0)

workers = 2
batch_size = 128
image_size = 28
nc = 1      # 이미지의 채널 수로, MNIST는 흑백이기 때문에 1. RGB 이미지는 3 설정
nz = 100    # 잠재공간 벡터의 크기
ngf = 28    # generator를 통과하는 feature map 크기
ndf = 28    # discriminator를 통과하는 feature map 크기

num_epochs = 200
lr = 0.0002

# ``netG`` 와 ``netD`` 에 적용시킬 커스텀 가중치 초기화 함수
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # 입력: (batch_size, nz, 1, 1) → 잠재 벡터 z
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 출력 크기: (batch_size, ngf*8, 4, 4)

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 출력 크기: (batch_size, ngf*4, 8, 8)

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 출력 크기: (batch_size, ngf*2, 16, 16)

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 출력 크기: (batch_size, ngf, 32, 32)

            nn.ConvTranspose2d(ngf, nc, kernel_size=1, stride=1, padding=2, bias=False),
            nn.Tanh()  # 픽셀 값을 [-1, 1] 범위로 정규화
            # 최종 출력 크기: (batch_size, nc, 28, 28)
        )
    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            # 입력: (batch_size, nc, 28, 28) → 입력 이미지는 28x28 크기
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 출력 크기: (batch_size, ndf, 32, 32)

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 출력 크기: (batch_size, ndf*2, 16, 16)

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 출력 크기: (batch_size, ndf*4, 8, 8)

            nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()  # 0~1 확률값 출력
            # 최종 출력 크기: (batch_size, 1, 1, 1) → (batch_size, 1)로 reshape
        )
    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)  # 차원 축소하여 (batch_size,) 형태로 반환



D = Discriminator().to(device)
G = Generator().to(device)


criterion = nn.BCELoss()
optimizer_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

# 데이터셋 불러오기
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

# 생성자의 학습상태를 확인할 latent space vector를 생성
fixed_noise = torch.randn(100, 100, 1, 1).to(device)

# 학습에 사용되는 참/거짓의 라벨
real_label = 1.
fake_label = 0.

img_list = []
G_losses = []
D_losses = []
iters = 0
# 학습 루프
for epoch in range(num_epochs):
    for i, data in enumerate(trainloader):
        ############################
        # (1) D 신경망을 업데이트 합니다: log(D(x)) + log(1 - D(G(z)))를 최대화 합니다
        ###########################
        ## 진짜 데이터들로 학습을 합니다
        D.zero_grad()
        # 배치들의 사이즈나 사용할 디바이스에 맞게 조정합니다
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label,
                           dtype=torch.float, device=device)
        # 진짜 데이터들로 이루어진 배치를 D에 통과시킵니다
        output = D(real_cpu).view(-1)
        # 손실값을 구합니다
        errD_real = criterion(output, label)
        # 역전파의 과정에서 변화도를 계산합니다
        errD_real.backward()
        D_x = output.mean().item()

        ## 가짜 데이터들로 학습을 합니다
        # 생성자에 사용할 잠재공간 벡터를 생성합니다
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # G를 이용해 가짜 이미지를 생성합니다
        fake = G(noise)
        label.fill_(fake_label)
        # D를 이용해 데이터의 진위를 판별합니다
        output = D(fake.detach()).view(-1)
        # D의 손실값을 계산합니다
        errD_fake = criterion(output, label)
        # 역전파를 통해 변화도를 계산합니다. 이때 앞서 구한 변화도에 더합니다(accumulate)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # 가짜 이미지와 진짜 이미지 모두에서 구한 손실값들을 더합니다
        # 이때 errD는 역전파에서 사용되지 않고, 이후 학습 상태를 리포팅(reporting)할 때 사용합니다
        errD = errD_real + errD_fake
        # D를 업데이트 합니다
        optimizer_D.step()

        ############################
        # (2) G 신경망을 업데이트 합니다: log(D(G(z)))를 최대화 합니다
        ###########################
        G.zero_grad()
        label.fill_(real_label)  # 생성자의 손실값을 구하기 위해 진짜 라벨을 이용할 겁니다
        # 우리는 방금 D를 업데이트했기 때문에, D에 다시 가짜 데이터를 통과시킵니다.
        # 이때 G는 업데이트되지 않았지만, D가 업데이트 되었기 때문에 앞선 손실값가 다른 값이 나오게 됩니다
        output = D(fake).view(-1)
        # G의 손실값을 구합니다
        errG = criterion(output, label)
        # G의 변화도를 계산합니다
        errG.backward()
        D_G_z2 = output.mean().item()
        # G를 업데이트 합니다
        optimizer_G.step()

        # 훈련 상태를 출력
        if i % 100 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(trainloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

    from torchvision.utils import save_image
    with torch.no_grad():
        save_path= "/home/ivpl-d29/myProject/GenerativeAI/GAN"
        save_image(G(fixed_noise), f'DCGAN_results/DCGAN_fake_samples{epoch+1}.png', nrow=10, normalize=True)


