import torch

from model import *
from utils import *

import matplotlib
import matplotlib.pyplot as plt
#import seaborn as sns
import math
import numpy as np


g_optim = 'newton'
d_optim = 'newton'
epochs = 1000
step = 2

def comp_gen2(g_optim, d_optim):
    dist = np.array([0.]*int(epochs/step))
    for i in range(step, epochs + step, step):
        dist0 = torch.load("./checkpoints/mnist/{}-1.0-{}-1.0-1/generator-epoch_{:d}.tar".format(g_optim, d_optim, i), map_location='cpu')['gradient']
        dist[int(i/step) - 1] = norm(dist0)
    return dist

def comp_dis(g_optim, d_optim):
    dist = np.array([0.]*int(epochs/step))
    for i in range(step, epochs + step, step):
        dist0 = torch.load("./checkpoints/mnist/{}-1.0-{}-1.0-1/discriminator-epoch_{:d}.tar".format(g_optim, d_optim, i), map_location='cpu')['gradient']
        dist[int(i/step) - 1] = norm(dist0)
    return dist


if __name__ == "__main__":
    epochs = 1000
    interval = 2
    dist1 = comp_gen2(g_optim, d_optim)[:int(epochs/interval)]
    print('generator', dist1)
    np.save(g_optim + d_optim + '_gen.npy', dist1)
    #dist2 = comp_gen2('gd', 'newton')[:int(epochs/interval)]
    #dist3 = comp_gen2('sd', 'gd')[:int(epochs/interval)]
    #dist4 = comp_gen2('gd', 'fr')[:int(epochs/interval)]

    ddist1 = comp_dis(g_optim, d_optim)[:int(epochs/interval)]
    #ddist2 = comp_dis('gd', 'newton')[:int(epochs/interval)]
    #ddist3 = comp_dis('sd', 'gd')[:int(epochs/interval)]
    #ddist4 = comp_dis('gd', 'fr')[:int(epochs/interval)]
    print('discriminator', ddist1)
    np.save(g_optim + d_optim + '_dis.npy', ddist1)
    ax1 = plt.subplot(121)
    ax1.plot(range(0, epochs, interval), dist1, label='gda', linestyle='-')
    #ax1.plot(range(0, epochs, interval), dist2, label='gdn', linestyle='--')
    #ax1.plot(range(0, epochs, interval), dist3, label='sd', linestyle=':')
    #ax1.plot(range(0, epochs, interval), dist4, label='fr', linestyle='-.')
    ax1.legend()
    plt.yscale('log')
    ax1.set_title('$generator gradient norm$')
    ax2 = plt.subplot(122)
    ax2.plot(range(0, epochs, interval), ddist1, label='gda', linestyle='-')
    #ax2.plot(range(0, epochs, interval), ddist2, label='gdn', linestyle='--')
    #ax2.plot(range(0, epochs, interval), ddist3, label='sd', linestyle=':')
    #ax2.plot(range(0, epochs, interval), ddist4, label='fr', linestyle='-.')
    ax2.legend()
    ax2.set_title('$discriminator gradient norm$')
    plt.yscale('log')
    plt.savefig('images/gd-gd.png')
    plt.show()

    #generator.load_state_dict(torch.load("./checkpoints/covariance/gd-newton/generator-epoch_1000.tar", map_location='cpu')['model_state_dict'])

    #noise = torch.randn(10000, 2).double()
    #test_pt = generator(noise).detach().numpy()

    #sns.kdeplot(test_pt[:, 0], test_pt[:, 1], shade='True')
    # plt.savefig('images/' + 'gmm.png')
    #plt.show()

    # i = 10
    # discriminator = torch.load("./checkpoints/discriminator-epoch_{:03d}.pth".format(i), map_location='cpu').eval()
    # generator = torch.load("./checkpoints/generator-epoch_{:03d}.pth".format(i), map_location='cpu').eval()

    # dataset = torch.randn(30, 2)
    # dataset = torch.utils.data.TensorDataset(dataset)

    # loader = torch.utils.data.DataLoader(dataset, batch_size=30, shuffle=True)

    # plot_visualization(discriminator, generator, loader)
    # plt.show()



    # def get_lst(g_optim, d_optim):
    #     pattern = "./checkpoints/single_gaussian/{}-{}/{}-epoch_{:03d}.tar"

    #     lst_eta = []
    #     lst_w = []


    #     lst_epoch = np.arange(0, 1000, 20)

    #     for i in lst_epoch:
    #         discriminator = OneLayerNet(2)
    #         generator = ShiftNet(2)

    #         ckpt = torch.load(pattern.format(g_optim, d_optim, "discriminator", i), map_location='cpu')
    #         discriminator.load_state_dict(ckpt['model_state_dict'])

    #         ckpt = torch.load(pattern.format(g_optim, d_optim, "generator", i), map_location='cpu')
    #         generator.load_state_dict(ckpt['model_state_dict'])

    #         lst_eta.append(generator.get_numpy_eta())
    #         lst_w.append(discriminator.w.detach().numpy())

    #     return lst_epoch, lst_eta, lst_w

    # lst_epoch, lst_gd_gd_eta, lst_gd_gd_w = get_lst("gd", "gd")
    # lst_epoch, lst_sd_gd_eta, lst_sd_gd_w = get_lst("sd", "gd")
    # lst_epoch, lst_gd_fr_eta, lst_gd_fr_w = get_lst("gd", "fr")
    # lst_epoch, lst_gd_newton_eta, lst_gd_newton_w = get_lst("gd", "newton")

    # fig, axes = plt.subplots(figsize=(6, 3), nrows=1, ncols=2)
    # axes[0].plot(lst_epoch, [np.linalg.norm(xx) for xx in lst_gd_gd_eta], linestyle='-', label='gda')
    # axes[0].plot(lst_epoch, [np.linalg.norm(xx) for xx in lst_sd_gd_eta], linestyle='--', label='sd')
    # axes[0].plot(lst_epoch, [np.linalg.norm(xx) for xx in lst_gd_fr_eta], linestyle=':', label='fr')
    # axes[0].plot(lst_epoch, [np.linalg.norm(xx) for xx in lst_gd_newton_eta], linestyle='-.', label='gdn')
    # axes[0].set_yscale('log')
    # axes[0].set_xlabel("epoch")
    # axes[0].set_ylabel(r"$\vert| \eta \vert|$")
    # axes[0].legend(loc='lower left')

    # axes[1].plot(lst_epoch, [np.linalg.norm(xx) for xx in lst_gd_gd_w], linestyle='-', label='gda')
    # axes[1].plot(lst_epoch, [np.linalg.norm(xx) for xx in lst_sd_gd_w], linestyle='--', label='sd')
    # axes[1].plot(lst_epoch, [np.linalg.norm(xx) for xx in lst_gd_fr_w], linestyle=':', label='fr')
    # axes[1].plot(lst_epoch, [np.linalg.norm(xx) for xx in lst_gd_newton_w], linestyle='-.', label='gdn')
    # axes[1].set_yscale('log')
    # axes[1].set_xlabel("epoch")
    # axes[1].set_ylabel(r"$\vert| \omega \vert|$")
    # # axes[1].legend(loc='upper right')
    # axes[1].legend(loc='lower left')

    # plt.tight_layout()

    # plt.show()
