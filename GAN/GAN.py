"""
godestone님의 github code 참고
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
dir_name = "GAN_results"

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

# MNIST dataset setting, 데이터로더
MNIST_dataset = torchvision.datasets.MNIST(root='../data/', train=True, transform=transform, download=True)
data_loader = torch.utils.data.DataLoader(dataset=MNIST_dataset, batch_size=batch_size, shuffle=True)

# 판별자(Discriminator) 클래스 정의
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.linear1 = nn.Linear(img_size, hidden_size3)        # 28(이미지크기), 1024
        self.linear2 = nn.Linear(hidden_size3, hidden_size2)    # 1024, 512
        self.linear3 = nn.Linear(hidden_size2, hidden_size1)    # 512, 256
        self.linear4 = nn.Linear(hidden_size1, 1)    # 256, 1(흑백)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.leaky_relu(self.linear1(x))
        x = self.leaky_relu(self.linear2(x))
        x = self.leaky_relu(self.linear3(x))
        x = self.linear4(x)
        x = self.sigmoid(x)
        return x

# 생성자(Generator) 클래스 정의
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(noise_size, hidden_size1)      # 100(노이즈크기), 256
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)    # 256, 512
        self.linear3 = nn.Linear(hidden_size2, hidden_size3)    # 512, 1024
        self.linear4 = nn.Linear(hidden_size3, img_size)        # 1024, 28(이미지크기)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.linear4(x)
        x = self.tanh(x)
        return x


# 생성자와 판별자 초기화
discriminator = Discriminator()
generator = Generator()

# GPU로 이동
discriminator = discriminator.to(device)
generator = generator.to(device)

# Loss function & Optimizer
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)


"""
Training part
"""
for epoch in range(num_epoch):
    for i, (images, label) in enumerate(data_loader):

        real_label = torch.ones((batch_size, 1), dtype=torch.float32).to(device)  # 진짜 라벨
        fake_label = torch.zeros((batch_size, 1), dtype=torch.float32).to(device) # 가짜 라벨

        real_images = images.reshape(batch_size, -1).to(device)  # 배치 차원 변경

        # +----------------------+
        # | Train Discriminator  |
        # +----------------------+

        d_optimizer.zero_grad()

        # Generator가 만든 가짜 이미지 생성
        z = torch.randn(batch_size, noise_size).to(device)
        fake_images = generator(z).detach()  # Discriminator 학습 시 G의 그래디언트 차단

        # D가 진짜와 가짜를 얼마나 잘 구분하는지 학습
        real_loss = criterion(discriminator(real_images), real_label)
        fake_loss = criterion(discriminator(fake_images), fake_label)
        d_loss = real_loss + fake_loss  # 합 사용

        # 역전파
        d_loss.backward()
        d_optimizer.step()

        # +------------------+
        # | Train Generator  |
        # +------------------+

        g_optimizer.zero_grad()

        # 새로운 가짜 이미지 생성
        z = torch.randn(batch_size, noise_size).to(device)
        fake_images = generator(z)

        # G가 만든 이미지를 D가 진짜라고 속이도록 학습
        g_loss = criterion(discriminator(fake_images), real_label)

        # 역전파
        g_loss.backward()
        g_optimizer.step()

        # 학습 과정 출력
        if (i + 1) % 150 == 0:
            print("Epoch [{}/{}] Step [{}/{}] d_loss: {:.5f} g_loss: {:.5f}"
                  .format(epoch, num_epoch, i+1, len(data_loader), d_loss.item(), g_loss.item()))

    # 각 epoch마다 D의 정확도를 측정하여 출력
    real_acc = (discriminator(real_images) > 0.5).float().mean().item()
    fake_acc = (discriminator(fake_images) < 0.5).float().mean().item()
    print("Epoch {}: Discriminator Real Acc: {:.2f} Fake Acc: {:.2f}".format(epoch, real_acc, fake_acc))

    # 각 epoch마다 생성된 가짜 이미지를 저장
    samples = fake_images.reshape(batch_size, 1, 28, 28)
    save_image(samples, os.path.join(dir_name, 'GAN_fake_samples{}.png'.format(epoch + 1)))
