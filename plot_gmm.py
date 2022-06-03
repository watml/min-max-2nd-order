import numpy as np

import torch
import torchvision

from model import *
from utils import *

import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

string = "gd-0.01-newton-1.0-1"
#string = "gd-0.01-newton-1.0-1"
# string = "gd-0.01-fr-0.02-1"
#string = "sd-0.01-gd-0.02-1"
# string = "pretrain"
pattern = "./checkpoints/gmm/{}/{}-epoch_{}.tar"
epoch = 1000

import argparse
import matplotlib
matplotlib.use('TkAgg')
def plot_db_discriminator(discriminator, x_min, x_max, y_min, y_max):
    device = 'cpu'

    ep = 1
    xx, yy = np.meshgrid(np.arange(x_min - ep, x_max + ep, 0.1),
                         np.arange(y_min - ep, y_max + ep, 0.1))

    points = np.column_stack((xx.ravel(), yy.ravel()))
    with torch.no_grad():
        zz = discriminator(torch.tensor(
            points, dtype=torch.double, device=device)).detach().cpu().numpy()
    zz = zz.reshape(xx.shape)

    print(zz)
    cp = plt.contourf(xx, yy, zz, levels=np.linspace(0.45, 0.55, 6))
    #cp = plt.contourf(xx, yy, zz, levels=np.linspace(0.0, 0.7, 8))
    plt.colorbar(cp)


discriminator = DNet().double()
discriminator.load_state_dict(torch.load(pattern.format(string, "discriminator", epoch), map_location='cpu')['model_state_dict'])

plt.figure(figsize=(4, 3))

plot_db_discriminator(lambda xx: discriminator(xx).sigmoid(), -2, 2, -2, 2)
plt.tight_layout()

generator = GNet().double()
generator.load_state_dict(torch.load(pattern.format(string, "generator", epoch), map_location='cpu')['model_state_dict'])

noise = torch.randn(50000, 100).double()
fake_data = generator(noise).detach().numpy()

plt.figure(figsize=(4, 3))

sns.kdeplot(fake_data[:, 0], fake_data[:, 1], shade=True, fill=True)

plt.show()
#plt.savefig()
