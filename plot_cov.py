import math

import numpy as np

import torch

from model import *

import matplotlib.pyplot as plt


def comp_gen(g_optim, d_optim):
    generator = AffineNet().double()
    sigma = torch.tensor([[1., 0.], [0., 0.04]], dtype=torch.double)
    dist = np.array([0.]*200)
    v = generator.V
    for i in range(1, 2001, 10):
        generator.load_state_dict(torch.load("./checkpoints/covariance/{:s}-{:s}/generator-epoch_{:d}.tar".format(g_optim, d_optim, i), map_location='cpu')['model_state_dict'])
        diff = v.t().mm(v).clone().detach() - sigma
        dist[int(i/10)] = (diff ** 2).sum()
    return dist


def comp_gen2(g_optim, d_optim):
    dist = np.array([0.]*200)
    for i in range(1, 2001, 10):
        dist0 = torch.load("./checkpoints/covariance/{:s}-{:s}/generator-epoch_{:d}.tar".format(g_optim, d_optim, i), map_location='cpu')['generator_norm']
        dist[int(i/10)] = dist0
    return dist


def comp_dis(g_optim, d_optim):
    discriminator = QuadraticNet(2).double()
    dist = np.array([0.]*200)
    for i in range(1, 2001, 10):
        discriminator.load_state_dict(torch.load("./checkpoints/covariance/{:s}-{:s}/discriminator-epoch_{:d}.tar".format(g_optim, d_optim, i), map_location='cpu')['model_state_dict'])
        w = discriminator.W
        w = (w + w.t())/2
        dist[int(i/10)] = math.sqrt((w ** 2).sum())
    return dist


if __name__ == "__main__":
    epochs = 2000
    # dist1 = comp_gen2('gd', 'gd')[:int(epochs/10)]
    dist2 = comp_gen2('gd', 'newton')[:int(epochs/10)]
    # dist3 = comp_gen2('sd', 'gd')[:int(epochs/10)]
    dist4 = comp_gen2('gd', 'fr')[:int(epochs/10)]

    # ddist1 = comp_dis('gd', 'gd')[:int(epochs/10)]
    ddist2 = comp_dis('gd', 'newton')[:int(epochs/10)]
    # ddist3 = comp_dis('sd', 'gd')[:int(epochs/10)]
    ddist4 = comp_dis('gd', 'fr')[:int(epochs/10)]
    ax1 = plt.subplot(121)
    # ax1.plot(range(0, epochs, 10), dist1, label='gda', linestyle='-')
    ax1.plot(range(0, epochs, 10), dist2, label='gdn', linestyle='--')
    # print(len(dist3))
    # ax1.plot(range(0, epochs, 10), dist3, label='sd', linestyle=':')
    ax1.plot(range(0, epochs, 10), dist4, label='fr', linestyle='-.')
    ax1.legend()
    plt.yscale('log')
    ax1.set_title(r'$||VV^T - \Sigma||_2$')
    ax2 = plt.subplot(122)
    # ax2.plot(range(0, epochs, 10), ddist1, label='gda', linestyle='-')
    ax2.plot(range(0, epochs, 10), ddist2, label='gdn', linestyle='--')
    # ax2.plot(range(0, epochs, 10), ddist3, label='sd', linestyle=':')
    ax2.plot(range(0, epochs, 10), ddist4, label='fr', linestyle='-.')
    ax2.legend()
    ax2.set_title(r'$||(W + W^T)/2||_2$')
    plt.yscale('log')
    plt.show()
    # plt.savefig('images/gd-gd.png')
