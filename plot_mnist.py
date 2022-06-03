import numpy as np

import torch
import torchvision

from model import *
from utils import *

import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

g_optim = 'newton'
d_optim = 'newton'
epoch=50

pattern = "./checkpoints/mnist/{}-1.0-{}-1.0-1/{}-epoch_{}.tar"
#pattern = "./checkpoints/mnist/{}-epoch_{}.tar"

generator = Generator().double()
generator.load_state_dict(torch.load(pattern.format(g_optim, d_optim, "generator", epoch), map_location='cpu')['model_state_dict'])

noise = torch.randn(25, 100).double()
fake_data = vectors2images(generator(noise)).detach()

grid = torchvision.utils.make_grid(vectors2images(fake_data), nrow=5, normalize=True, range=(-1, 1), padding=0)

fig = plt.figure(figsize=(4, 4))
grid = grid.numpy()
plt.imshow(np.transpose(grid, (1, 2, 0)), interpolation='nearest')
plt.axis('off')

plt.tight_layout()
plt.show()
