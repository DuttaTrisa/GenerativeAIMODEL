import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Define the generator network
class Generator(nn.Module):
    def _init_(self):
        super(Generator, self)._init_()
        self.fc1 = nn.Linear(100, 128)  # input layer (100) -> hidden layer (128)
        self.fc2 = nn.Linear(128, 256)  # hidden layer (128) -> hidden layer (256)
        self.fc3 = nn.Linear(256, 28*28)  # hidden layer (256) -> output layer (784)

    def forward(self, z):
        z = torch.relu(self.fc1(z))  # activation function for hidden layer
        z = torch.relu(self.fc2(z))
        z = torch.tanh(self.fc3(z))  # output layer with tanh activation
        return z.view(-1, 1, 28, 28)  # reshape to image dimensions

# Define the discriminator network
class Discriminator(nn.Module):
    def _init_(self):
        super(Discriminator, self)._init_()
        self.fc1 = nn.Linear(28*28, 256)  # input layer (784) -> hidden layer (256)
        self.fc2 = nn.Linear(256, 128)  # hidden layer (256) -> hidden layer (128)
        self.fc3 = nn.Linear(128, 1)  # hidden layer (128) -> output layer (1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # activation function for hidden layer
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # output layer with sigmoid activation
        return x

# Initialize the generator and discriminator
generator = Generator()
discriminator = Discriminator()

# Define the loss functions and optimizers
criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=0.001)
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.001)

# Create a dataset class for our simple images
class ImageDataset(Dataset):
    def _init_(self, size):
        self.size = size
        self.data = torch.randn(size, 1, 28, 28)

    def _len_(self):
        return self.size

    def _getitem_(self, idx):
        return self.data[idx]

# Create a data loader for our dataset
dataset = ImageDataset(1000)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train the GAN
for epoch in range(100):
    for i, real_images in enumerate(data_loader):
        # Sample a batch of random noise vectors
        z = torch.randn(32, 100)

        # Generate fake images using the generator
        fake_images = generator(z)

        # Train the discriminator
        optimizer_d.zero_grad()
        real_labels = torch.ones(32, 1)
        fake_labels = torch.zeros(32, 1)
        real_output = discriminator(real_images)
        fake_output = discriminator(fake_images)
        d_loss = criterion(real_output, real_labels) + criterion(fake_output, fake_labels)
        d_loss.backward()
        optimizer_d.step()

        # Train the generator
        optimizer_g.zero_grad()
        g_loss = criterion(fake_output, real_labels)
        g_loss.backward()
        optimizer_g.step()

    # Print the losses at each epoch
    print(f'Epoch {epoch+1}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')

# Generate some sample images using the trained generator
z = torch.randn(1, 100)
generated_image = generator(z)
plt.imshow(generated_image.detach().numpy().reshape(28, 28), cmap='gray')
plt.show()