"""
godestone님의 github code 참고

Generator 한 번 학습하지만 Discriminator는 k>1번 학습하여 discriminator 최적화 유도 방법
"""
import os
import torch.nn as nn
import torch.utils.data
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

# Hyper-parameters & Variables setting
num_epoch = 200
batch_size = 100
learning_rate = 0.0002
img_size = 28 * 28
num_channel = 1
dir_name = "GAN_k_step_results"
k_steps = 3  # Discriminator 학습 반복 횟수

noise_size = 100
hidden_size1 = 256
hidden_size2 = 512
hidden_size3 = 1024

# Device setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Now using {} devices".format(device))

# Create a directory for saving samples
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

# Dataset transform setting
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)])

# MNIST dataset setting
MNIST_dataset = torchvision.datasets.MNIST(root='../data/', train=True, transform=transform, download=True)
data_loader = torch.utils.data.DataLoader(dataset=MNIST_dataset, batch_size=batch_size, shuffle=True)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.linear1 = nn.Linear(img_size, hidden_size3)
        self.linear2 = nn.Linear(hidden_size3, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, hidden_size1)
        self.linear4 = nn.Linear(hidden_size1, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leaky_relu(self.linear1(x))
        x = self.leaky_relu(self.linear2(x))
        x = self.leaky_relu(self.linear3(x))
        x = self.linear4(x)
        x = self.sigmoid(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(noise_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, hidden_size3)
        self.linear4 = nn.Linear(hidden_size3, img_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.linear4(x)
        x = self.tanh(x)
        return x


def train_discriminator(discriminator, generator, real_images, d_optimizer, criterion, k_steps):
    """
    Discriminator를 k번 학습시키는 함수
    """
    d_losses = []
    real_accs = []
    fake_accs = []

    for _ in range(k_steps):    # 학습을 k-steps번 반복함
        d_optimizer.zero_grad()

        # 진짜 이미지에 대한 판별
        real_label = torch.ones((real_images.size(0), 1)).to(device)
        real_output = discriminator(real_images)
        real_loss = criterion(real_output, real_label)

        # 가짜 이미지 생성 및 판별
        z = torch.randn(real_images.size(0), noise_size).to(device)
        fake_images = generator(z).detach()
        fake_label = torch.zeros((real_images.size(0), 1)).to(device)
        fake_output = discriminator(fake_images)
        fake_loss = criterion(fake_output, fake_label)

        # 전체 손실 계산 및 역전파
        d_loss = real_loss + fake_loss
        d_loss.backward()
        d_optimizer.step()

        # 정확도 계산
        real_acc = (real_output > 0.5).float().mean().item()
        fake_acc = (fake_output < 0.5).float().mean().item()

        d_losses.append(d_loss.item())
        real_accs.append(real_acc)
        fake_accs.append(fake_acc)

    return sum(d_losses) / len(d_losses), sum(real_accs) / len(real_accs), sum(fake_accs) / len(fake_accs)


# 모델 초기화
discriminator = Discriminator().to(device)
generator = Generator().to(device)

# Loss function & Optimizer
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)

# Training
for epoch in range(num_epoch):
    for i, (images, _) in enumerate(data_loader):
        real_images = images.reshape(batch_size, -1).to(device)

        # Train Discriminator k steps
        d_loss, real_acc, fake_acc = train_discriminator(
            discriminator, generator, real_images, d_optimizer, criterion, k_steps
        )

        # Train Generator
        g_optimizer.zero_grad()
        z = torch.randn(batch_size, noise_size).to(device)
        fake_images = generator(z)
        real_label = torch.ones((batch_size, 1)).to(device)
        g_loss = criterion(discriminator(fake_images), real_label)
        g_loss.backward()
        g_optimizer.step()

        # 학습 과정 출력
        if (i + 1) % 150 == 0:
            print(f"Epoch [{epoch}/{num_epoch}] Step [{i + 1}/{len(data_loader)}] "
                  f"d_loss: {d_loss:.5f} g_loss: {g_loss.item():.5f}")

    # 각 epoch마다 D의 정확도 출력
    print(f"Epoch {epoch}: Discriminator Real Acc: {real_acc:.2f} Fake Acc: {fake_acc:.2f}")

    # 가짜 이미지 저장
    samples = fake_images.reshape(batch_size, 1, 28, 28)
    save_image(samples, os.path.join(dir_name, f'GAN_{k_steps}steps_fake_samples{epoch + 1}.png'))