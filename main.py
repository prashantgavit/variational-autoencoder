import sys
import os
from torchvision.transforms import Compose, Resize, ToTensor
import torch

# from torchmeta.transforms import Categorical, ClassSplitter, Rotation

base_pth = os.getcwd().split('vae-omniglot')[0] + 'vae-omniglot/'

sys.path.append(base_pth)

from torch.utils.data import DataLoader

from models.cvae import ConditionalVAE

batch_size = 8
in_channels = 3
num_classes = 1
latent_dim = 20
img_size = 105
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ConditionalVAE(
    in_channels=in_channels,
    num_classes=num_classes,
    latent_dim=latent_dim,
    img_size=img_size
).to(device)

x = torch.randn(batch_size, in_channels, img_size, img_size).to(device)

labels = torch.zeros(batch_size, num_classes).to(device)
labels[:, 0] = 1  # Example: all samples belong to class 0
print('x before model', x.shape)
print('label before model', labels.shape)
output = model(x, labels=labels)
print("ConditionalVAE output shapes:")
print("Reconstruction:", output[0].shape)
print("Input:", output[1].shape)
print("Mu:", output[2].shape)
print("LogVar:", output[3].shape)