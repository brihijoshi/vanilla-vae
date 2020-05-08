"""
Reference - https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
"""

import numpy as np
import torch
from torch import nn
import torch.optim as opt
from torch.nn import functional as F

def conv_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv3D
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int))
    return outshape

class VanillaVAE(nn.Module):
    

    def __init__(self,in_channels, latent_dim):
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim
        
        output_shape = (28,28)
        
        self.enc1 = nn.Sequential(
                    nn.Conv2d(1, out_channels=32,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(32),
                    nn.LeakyReLU())
        output_shape1 = conv_output_size(output_shape,(1,1),(3,3),(2,2))

        self.enc2 = nn.Sequential(
                    nn.Conv2d(32, out_channels=64,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU())
        output_shape2 = conv_output_size(output_shape1,(1,1),(3,3),(2,2))
        
        self.enc3 = nn.Sequential(
                    nn.Conv2d(64, out_channels=128,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU())
        output_shape3 = conv_output_size(output_shape2,(1,1),(3,3),(2,2))
        
        self.enc4 = nn.Sequential(
                    nn.Conv2d(128, out_channels=256,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(256),
                    nn.LeakyReLU())
        output_shape4 = conv_output_size(output_shape3,(1,1),(3,3),(2,2))
        
        self.enc5 = nn.Sequential(
                    nn.Conv2d(256, out_channels=512,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(512),
                    nn.LeakyReLU())
        output_shape5 = conv_output_size(output_shape4,(1,1),(3,3),(2,2))
        
        
        self.fc_mu = nn.Linear(512*output_shape5[0]*output_shape5[1], latent_dim)
        self.fc_var = nn.Linear(512*output_shape5[0]*output_shape5[1], latent_dim)
        
        self.decoder_input = nn.Linear(latent_dim, 512*output_shape5[0]*output_shape5[1])
        
        self.dec5 = nn.Sequential(
                    nn.ConvTranspose2d(512, out_channels=256,
                              kernel_size= 3, stride= 2, padding  = 1, output_padding=1),
                    nn.BatchNorm2d(256),
                    nn.LeakyReLU())
        
        self.dec4 = nn.Sequential(
                    nn.ConvTranspose2d(256, out_channels=128,
                              kernel_size= 3, stride= 2, padding  = 1, output_padding=1),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU())
        
        self.dec3 = nn.Sequential(
                    nn.ConvTranspose2d(128, out_channels=64,
                              kernel_size= 3, stride= 2, padding  = 1, output_padding=1),
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU())
        
        self.dec2 = nn.Sequential(
                    nn.ConvTranspose2d(64, out_channels=32,
                              kernel_size= 3, stride= 2, padding  = 1, output_padding=1),
                    nn.BatchNorm2d(32),
                    nn.LeakyReLU())

        self.dec1 = nn.Sequential(
                    nn.ConvTranspose2d(32, out_channels=1,
                              kernel_size= 3, stride= 2, padding  = 1, output_padding=1))
        


    def encode(self, inp):
        result = self.enc1(inp)
        result = self.enc2(result)
        result = self.enc3(result)
        result = self.enc4(result)
        result = self.enc5(result)
        result = torch.flatten(result, start_dim=1)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 512, 1, 1)
        result = self.dec5(result)
        result = self.dec4(result)
        result = self.dec3(result)
        result = self.dec2(result)
        result = self.dec1(result)

        return result

    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, inp):
        mu, log_var = self.encode(inp)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), inp, mu, log_var]

    def sample(self,num_samples,current_device):
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x):

        return self.forward(x)[0]