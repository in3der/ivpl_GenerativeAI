"""
godestone님의 github code 참고

discriminator를 k<9번 정도 'simulate'하여 mode collapse 해결하는 방법
"""
import os
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import copy

# Hyper-parameters & Variables setting
num_epoch = 200
batch_size = 100
learning_rate = 0.0002
img_size = 28 * 28
num_channel = 1
dir_name = "UnrolledGAN_results"
unroll_steps = 3  # k steps for unrolling discriminator

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


def update_discriminator(discriminator, real_images, fake_images, criterion, optimizer):
    batch_size = real_images.size(0)
    real_label = torch.ones((batch_size, 1)).to(device)
    fake_label = torch.zeros((batch_size, 1)).to(device)

    optimizer.zero_grad()

    # Real images
    real_loss = criterion(discriminator(real_images), real_label)
    # Fake images
    fake_loss = criterion(discriminator(fake_images), fake_label)

    d_loss = real_loss + fake_loss
    d_loss.backward()
    optimizer.step()

    return d_loss


discriminator = Discriminator().to(device)
generator = Generator().to(device)

criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)

# Training
for epoch in range(num_epoch):
    for i, (images, _) in enumerate(data_loader):
        real_images = images.reshape(batch_size, -1).to(device)

        # Train Discriminator
        z = torch.randn(batch_size, noise_size).to(device)
        fake_images = generator(z).detach()
        d_loss = update_discriminator(discriminator, real_images, fake_images, criterion, d_optimizer)

        # Train Generator with unrolling
        g_optimizer.zero_grad()

        # Store original discriminator parameters
        stored_parameters = copy.deepcopy(discriminator.state_dict())

        # Simulate k steps of discriminator
        z = torch.randn(batch_size, noise_size).to(device)
        surrogate_fake_images = generator(z)

        for _ in range(unroll_steps):
            update_discriminator(discriminator, real_images, surrogate_fake_images.detach(),
                                 criterion, d_optimizer)

        # Generator loss with k-step discriminator
        z = torch.randn(batch_size, noise_size).to(device)
        fake_images = generator(z)
        real_label = torch.ones((batch_size, 1)).to(device)
        g_loss = criterion(discriminator(fake_images), real_label)

        # Generator backward pass
        g_loss.backward()
        g_optimizer.step()

        # Restore discriminator parameters
        discriminator.load_state_dict(stored_parameters)

        # Print progress
        if (i + 1) % 150 == 0:
            print(f"Epoch [{epoch}/{num_epoch}] Step [{i + 1}/{len(data_loader)}] "
                  f"d_loss: {d_loss.item():.5f} g_loss: {g_loss.item():.5f}")

    # Measure discriminator accuracy per epoch
    with torch.no_grad():
        real_acc = (discriminator(real_images) > 0.5).float().mean().item()
        fake_acc = (discriminator(fake_images) < 0.5).float().mean().item()
        print(f"Epoch {epoch}: Discriminator Real Acc: {real_acc:.2f} Fake Acc: {fake_acc:.2f}")

    # Save generated images
    samples = fake_images.reshape(batch_size, 1, 28, 28)
    save_image(samples, os.path.join(dir_name, f'UnrolledGAN_{unroll_steps}steps_samples{epoch + 1}.png'))