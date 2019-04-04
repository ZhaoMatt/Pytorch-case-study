import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from ImageDataset import ImageDataset
from torch.utils.data.sampler import SubsetRandomSampler

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

# Dataset and Dataloader
transform = transforms.Compose([
    transforms.Resize(128)
])
dataset = ImageDataset('../data/celeba', transform=transform)

batch_size = 16

data_loader = loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)


# Model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__

        def block(in_feat, out_feat, normalize = True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            block(opt.latent_dim, 128, normalize=True)
            block(128, 256)
            block(256,512)
            block(512, 1024)
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
    def forward(self,z):
        img = self.model(z)
        img = img.view(img.size(0), img_shape)
        return img

class Discrimator(nn.Module):
    def __init__(slef):
        super(Discrimator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0),-1)
        validity = self.model(img_flat)
        return validity
    
# Loss Function
adversarial_loss = torch.nn.BCELoss() #Binary Cross Entropy

# initialize generator and descriminator
generator = Generator()
discriminator = Discrimator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# optimizer
optimizer_G = torch.optim.Adam(generator.parameters, lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.otim.Adam(discriminator.parameters, lr=opt.lr, betas=(opt.b1, opt.b2))

# Tensor = torch.cuda.FloatTen

# Training
for epoch in range(opt.n_epochs):
