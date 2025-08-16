import torch
import torch.nn as nn

import numpy as np

from tqdm import tqdm
from torchvision.utils import save_image, make_grid


def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + KLD

class Encoder(nn.Module):
    def __init__(self, input_channels, latent_dim, image_size=32):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        # Dynamically compute flattened feature size
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, image_size, image_size)
            h = self.LeakyReLU(self.conv1(dummy))
            h = self.LeakyReLU(self.conv2(h))
            h = self.LeakyReLU(self.conv3(h))
            h = self.LeakyReLU(self.conv4(h))
            flat_dim = h.numel()
        self.fc_mean = nn.Linear(flat_dim, latent_dim)
        self.fc_var = nn.Linear(flat_dim, latent_dim)

    def forward(self, x):
        h = self.LeakyReLU(self.conv1(x))
        h = self.LeakyReLU(self.conv2(h))
        h = self.LeakyReLU(self.conv3(h))
        h = self.LeakyReLU(self.conv4(h))
        h = self.flatten(h)
        mean = self.fc_mean(h)
        log_var = self.fc_var(h)
        return mean, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_channels):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 256 * 2 * 2)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1)
        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = self.fc(x)
        h = h.view(-1, 256, 2, 2)
        h = self.LeakyReLU(self.deconv1(h))
        h = self.LeakyReLU(self.deconv2(h))
        h = self.LeakyReLU(self.deconv3(h))
        x_hat = torch.sigmoid(self.deconv4(h))
        return x_hat
    

class Model(nn.Module):
    def __init__(self, Encoder, Decoder,device):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        self.device = device
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)        # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
                
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat = self.Decoder(z)
        return x_hat, mean, log_var


if __name__ == "__main__":
    # Example input: batch of 16 grayscale images, 1x32x32
    batch_size = 16
    input_channels = 1
    image_size = 32
    latent_dim = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder(input_channels, latent_dim).to(device)
    decoder = Decoder(latent_dim, input_channels).to(device)
    model = Model(encoder, decoder, device).to(device)

    x = torch.randn(batch_size, input_channels, image_size, image_size).to(device)
    x_hat, mean, log_var = model(x)
    print(f"x_hat shape: {x_hat.shape}")
    print(f"mean shape: {mean.shape}")
    print(f"log_var shape: {log_var.shape}")
    
    # Test Decoder independently
    print("\nTesting Decoder independently:")
    latent_dim = 20
    output_channels = 1
    batch_size = 16
    image_size = 32
    decoder = Decoder(latent_dim, output_channels)
    z = torch.randn(batch_size, latent_dim)
    x_hat = decoder(z)
    print(f"Decoder output shape: {x_hat.shape}")
    expected_shape = (batch_size, output_channels, image_size, image_size)
    print(f"Expected shape: {expected_shape}")
    print("Shape match:", x_hat.shape == expected_shape)

