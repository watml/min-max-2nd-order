import numpy as np

import torch
import torchvision

from model import *

from utils import *

import matplotlib.pyplot as plt
plt.rc('pdf', fonttype=42)


def plot_single_gaussian():
    def get_lst(g_optim, g_step_size, d_optim, d_step_size, d_num_step):
        # pattern = "./checkpoints/single_gaussian/{}-{}-{}-{}-{}/{}-epoch_{:d}.tar"
        pattern = "./checkpoints/single_gaussian_ill_conditioned/{}-{}-{}-{}-{}/{}-epoch_{:d}.tar"

        lst_eta = []
        lst_w = []

        lst_epoch = np.arange(1, 1001, 2)

        for i in lst_epoch:
            discriminator = OneLayerNet(2)
            generator = ShiftNet(2)

            ckpt = torch.load(pattern.format(g_optim, g_step_size, d_optim, d_step_size, d_num_step, "discriminator", i), map_location='cpu')
            discriminator.load_state_dict(ckpt['model_state_dict'])

            ckpt = torch.load(pattern.format(g_optim, g_step_size, d_optim, d_step_size, d_num_step, "generator", i), map_location='cpu')
            generator.load_state_dict(ckpt['model_state_dict'])

            lst_eta.append(generator.get_numpy_eta())
            lst_w.append(discriminator.w.detach().numpy())

        return lst_epoch, lst_eta, lst_w

    # lst_epoch, lst_gd_gd_eta, lst_gd_gd_w = get_lst("gd", 0.05, "gd", 0.5, 1)
    lst_epoch, lst_2ts_gd_gd_eta, lst_2ts_gd_gd_w = get_lst("gd", 0.05, "gd", 0.5, 1)
    lst_epoch, lst_gd_gd_unrolled_eta, lst_gd_gd_unrolled_w = get_lst("gd", 0.05, "gd", 0.05, 20)
    lst_epoch, lst_sd_gd_eta, lst_sd_gd_w = get_lst("sd", 0.05, "gd", 0.5, 1)
    lst_epoch, lst_gd_fr_eta, lst_gd_fr_w = get_lst("gd", 0.05, "fr", 0.5, 1)
    lst_epoch, lst_gd_newton_eta, lst_gd_newton_w = get_lst("gd", 0.05, "newton", 1.0, 1)
    lst_epoch, lst_newton_newton_eta, lst_newton_newton_w = get_lst("newton", 1.0, "newton", 1.0, 1)

    fig, axes = plt.subplots(figsize=(7.5, 3.5), nrows=1, ncols=2)
    # axes[0].plot(lst_epoch, [np.linalg.norm(xx) for xx in lst_gd_gd_eta], linewidth=1.5, linestyle='-', label='gda')
    axes[0].plot(lst_epoch, [np.linalg.norm(xx) for xx in lst_2ts_gd_gd_eta], linewidth=1.5, linestyle=':', label='2ts-gda', color='tab:red')
    axes[0].plot(lst_epoch, [np.linalg.norm(xx) for xx in lst_gd_gd_unrolled_eta], linewidth=1.5, linestyle=':', label='gda-20', color='tab:cyan')
    axes[0].plot(lst_epoch, [np.linalg.norm(xx) for xx in lst_sd_gd_eta], linewidth=2, linestyle='-.', label='tgda', color='tab:olive')
    axes[0].plot(lst_epoch, [np.linalg.norm(xx) for xx in lst_gd_fr_eta], linewidth=2, linestyle='-.', label='fr', color='tab:pink')
    axes[0].plot(lst_epoch, [np.linalg.norm(xx) for xx in lst_gd_newton_eta], linewidth=2, linestyle='--', label='gdn', color='tab:blue')
    axes[0].plot(lst_epoch, [np.linalg.norm(xx) for xx in lst_newton_newton_eta], linewidth=1, linestyle='-', label='cn', color='tab:orange')
    axes[0].set_yscale('log')
    axes[0].set_xlabel("epoch", fontsize=15)
    axes[0].set_ylabel(r"generator $\vert| \eta \vert|$", fontsize=15)
    axes[0].tick_params(labelsize=12)
    axes[0].legend(loc='center', bbox_to_anchor=(0.3, 0.4), fontsize=10)

    # axes[1].plot(lst_epoch, [np.linalg.norm(xx) for xx in lst_gd_gd_w], linewidth=1.5, linestyle='-', label='gda')
    axes[1].plot(lst_epoch, [np.linalg.norm(xx) for xx in lst_2ts_gd_gd_w], linewidth=1.5, linestyle=':', label='2ts-gda', color='tab:red')
    axes[1].plot(lst_epoch, [np.linalg.norm(xx) for xx in lst_gd_gd_unrolled_w], linewidth=1.5, linestyle=':', label='gda-20', color='tab:cyan')
    axes[1].plot(lst_epoch, [np.linalg.norm(xx) for xx in lst_sd_gd_w], linewidth=2, linestyle='-.', label='tgda', color='tab:olive')
    axes[1].plot(lst_epoch, [np.linalg.norm(xx) for xx in lst_gd_fr_w], linewidth=2, linestyle='-.', label='fr', color='tab:pink')
    axes[1].plot(lst_epoch, [np.linalg.norm(xx) for xx in lst_gd_newton_w], linewidth=2, linestyle='--', label='gdn', color='tab:blue')
    axes[1].plot(lst_epoch, [np.linalg.norm(xx) for xx in lst_newton_newton_w], linewidth=1, linestyle='-', label='cn', color='tab:orange')
    axes[1].set_yscale('log')
    axes[1].set_xlabel("epoch", fontsize=15)
    axes[1].set_ylabel(r"discrimiantor $\vert| \omega \vert|$", fontsize=15)
    axes[1].tick_params(labelsize=12)
    axes[1].legend(loc='center', bbox_to_anchor=(0.3, 0.4), fontsize=10)

    # fig.subplots_adjust(hspace=0.0, wspace=0.0)
    # fig.subplots_adjust(left=0.15, right=0.99, top=0.98, bottom=0.15)

    plt.tight_layout()

    plt.show()


def plot_covariance():
    def get_lst(g_optim, g_step_size, d_optim, d_step_size, d_num_step):
        pattern = "./checkpoints/covariance/{}-{}-{}-{}-{}/{}-epoch_{:d}.tar"

        lst_v_dist = []
        lst_w = []

        lst_epoch = np.arange(1, 2001, 2)

        for i in lst_epoch:
            discriminator = QuadraticNet(2)
            # generator = AffineNet(2)

            ckpt = torch.load(pattern.format(g_optim, g_step_size, d_optim, d_step_size, d_num_step, "discriminator", i), map_location='cpu')
            discriminator.load_state_dict(ckpt['model_state_dict'])
            lst_w.append(discriminator.W.detach().numpy())

            ckpt = torch.load(pattern.format(g_optim, g_step_size, d_optim, d_step_size, d_num_step, "generator", i), map_location='cpu')
            lst_v_dist.append(ckpt['generator_norm'])

        return lst_epoch, lst_v_dist, lst_w

    # lst_epoch, lst_gd_gd_eta, lst_gd_gd_w = get_lst("gd", 0.05, "gd", 0.5, 1)
    lst_epoch, lst_2ts_gd_gd_v, lst_2ts_gd_gd_w = get_lst("gd", 0.02, "gd", 0.2, 1)
    lst_epoch, lst_gd_gd_unrolled_v, lst_gd_gd_unrolled_w = get_lst("gd", 0.02, "gd", 0.02, 20)
    lst_epoch, lst_sd_gd_v, lst_sd_gd_w = get_lst("sd", 0.02, "gd", 0.2, 1)
    lst_epoch, lst_gd_fr_v, lst_gd_fr_w = get_lst("gd", 0.02, "fr", 0.2, 1)
    lst_epoch, lst_gd_newton_v, lst_gd_newton_w = get_lst("gd", 0.02, "newton", 1.0, 1)
    lst_epoch, lst_newton_newton_v, lst_newton_newton_w = get_lst("newton", 1.0, "newton", 1.0, 1)

    fig, axes = plt.subplots(figsize=(7.5, 3.5), nrows=1, ncols=2)
    # axes[0].plot(lst_epoch, [np.linalg.norm(xx) for xx in lst_gd_gd_eta], linewidth=1.5, linestyle='-', label='gda')
    axes[0].plot(lst_epoch, lst_2ts_gd_gd_v, linewidth=1.5, linestyle=':', label='2ts-gda', color='tab:red')
    axes[0].plot(lst_epoch, lst_gd_gd_unrolled_v, linewidth=1.5, linestyle=':', label='gda-20', color='tab:cyan')
    axes[0].plot(lst_epoch, lst_sd_gd_v, linewidth=2, linestyle='-.', label='tgda', color='tab:olive')
    axes[0].plot(lst_epoch, lst_gd_fr_v, linewidth=2, linestyle='-.', label='fr', color='tab:pink')
    axes[0].plot(lst_epoch, lst_gd_newton_v, linewidth=2, linestyle='--', label='gdn', color='tab:blue')
    axes[0].plot(lst_epoch, lst_newton_newton_v, linewidth=1, linestyle='-', label='cn', color='tab:orange')
    axes[0].set_yscale('log')
    axes[0].set_xlabel("epoch", fontsize=15)
    axes[0].set_ylabel(r'generator $||VV^\top - \Sigma||_\mathrm{F}$', fontsize=15)
    axes[0].tick_params(labelsize=12)
    axes[0].legend(loc='center', bbox_to_anchor=(0.3, 0.4), fontsize=10)

    # axes[1].plot(lst_epoch, [np.linalg.norm(xx) for xx in lst_gd_gd_w], linewidth=1.5, linestyle='-', label='gda')
    axes[1].plot(lst_epoch, [np.linalg.norm(0.5 * (xx + xx.T)) for xx in lst_2ts_gd_gd_w], linewidth=1.5, linestyle=':', label='2ts-gda', color='tab:red')
    axes[1].plot(lst_epoch, [np.linalg.norm(0.5 * (xx + xx.T)) for xx in lst_gd_gd_unrolled_w], linewidth=1.5, linestyle=':', label='gda-20', color='tab:cyan')
    axes[1].plot(lst_epoch, [np.linalg.norm(0.5 * (xx + xx.T)) for xx in lst_sd_gd_w], linewidth=2, linestyle='-.', label='tgda', color='tab:olive')
    axes[1].plot(lst_epoch, [np.linalg.norm(0.5 * (xx + xx.T)) for xx in lst_gd_fr_w], linewidth=2, linestyle='-.', label='fr', color='tab:pink')
    axes[1].plot(lst_epoch, [np.linalg.norm(0.5 * (xx + xx.T)) for xx in lst_gd_newton_w], linewidth=2, linestyle='--', label='gdn', color='tab:blue')
    axes[1].plot(lst_epoch, [np.linalg.norm(0.5 * (xx + xx.T)) for xx in lst_newton_newton_w], linewidth=1, linestyle='-', label='cn', color='tab:orange')
    axes[1].set_yscale('log')
    axes[1].set_xlabel("epoch", fontsize=15)
    axes[1].set_ylabel(r"discriminator $||\frac{1}{2} (W + W^\top)||_\mathrm{F}$", fontsize=14)
    axes[1].tick_params(labelsize=12)
    axes[1].legend(loc='center', bbox_to_anchor=(0.3, 0.4), fontsize=10)

    # fig.subplots_adjust(hspace=0.0, wspace=0.0)
    # fig.subplots_adjust(left=0.15, right=0.99, top=0.98, bottom=0.15)

    plt.tight_layout()

    plt.show()
    plt.savefig('images/single_gaussian_ill_conditioned.eps')


def plot_rmsprop_vs_newton():
    def get_lst(g_optim, g_step_size, d_optim, d_step_size):
        pattern = "./checkpoints/single_gaussian/{}-{}-{}-{}-1/{}-epoch_{:d}.tar"
        # pattern = "./checkpoints/single_gaussian_ill_conditioned/{}-{}-{}-{}-1/{}-epoch_{:d}.tar"

        lst_eta = []
        lst_w = []

        lst_epoch = np.arange(1, 1001, 2)

        for i in lst_epoch:
            discriminator = OneLayerNet(2)
            generator = ShiftNet(2)

            ckpt = torch.load(pattern.format(g_optim, g_step_size, d_optim, d_step_size, "discriminator", i), map_location='cpu')
            discriminator.load_state_dict(ckpt['model_state_dict'])

            ckpt = torch.load(pattern.format(g_optim, g_step_size, d_optim, d_step_size, "generator", i), map_location='cpu')
            generator.load_state_dict(ckpt['model_state_dict'])

            lst_eta.append(generator.get_numpy_eta())
            lst_w.append(discriminator.w.detach().numpy())

        return lst_epoch, lst_eta, lst_w

    # lst_epoch, lst_rmsprop_rmsprop_0001_eta, lst_rmsprop_rmsprop_0001_w = get_lst("rmsprop", 0.0001, "rmsprop", 0.0001)
    lst_epoch, lst_rmsprop_rmsprop_0005_eta, lst_rmsprop_rmsprop_0005_w = get_lst("rmsprop", 0.0005, "rmsprop", 0.0005)
    lst_epoch, lst_rmsprop_rmsprop_001_eta, lst_rmsprop_rmsprop_001_w = get_lst("rmsprop", 0.001, "rmsprop", 0.001)
    lst_epoch, lst_rmsprop_rmsprop_01_eta, lst_rmsprop_rmsprop_01_w = get_lst("rmsprop", 0.01, "rmsprop", 0.01)

    lst_epoch, lst_adam_adam_0005_eta, lst_adam_adam_0005_w = get_lst("adam", 0.0005, "adam", 0.0005)
    lst_epoch, lst_adam_adam_001_eta, lst_adam_adam_001_w = get_lst("adam", 0.001, "adam", 0.001)
    lst_epoch, lst_adam_adam_01_eta, lst_adam_adam_01_w = get_lst("adam", 0.01, "adam", 0.01)

    lst_epoch, lst_amsgrad_amsgrad_0005_eta, lst_amsgrad_amsgrad_0005_w = get_lst("amsgrad", 0.0005, "amsgrad", 0.0005)
    lst_epoch, lst_amsgrad_amsgrad_001_eta, lst_amsgrad_amsgrad_001_w = get_lst("amsgrad", 0.001, "amsgrad", 0.001)
    lst_epoch, lst_amsgrad_amsgrad_01_eta, lst_amsgrad_amsgrad_01_w = get_lst("amsgrad", 0.01, "amsgrad", 0.01)

    lst_epoch, lst_gd_newton_eta, lst_gd_newton_w = get_lst("gd", 0.05, "newton", 1.0)
    lst_epoch, lst_newton_newton_eta, lst_newton_newton_w = get_lst("newton", 1.0, "newton", 1.0)

    fig, axes = plt.subplots(figsize=(4.5, 4), nrows=1, ncols=1)
    # axes.plot(lst_epoch, [np.linalg.norm(xx) for xx in lst_rmsprop_rmsprop_0001_eta], linewidth=1, linestyle='-', label='rmsprop-0.0001')
    axes.plot(lst_epoch, [np.linalg.norm(xx) for xx in lst_rmsprop_rmsprop_0005_eta], linewidth=1, linestyle='-', label='rmsprop-0.0005')
    axes.plot(lst_epoch, [np.linalg.norm(xx) for xx in lst_rmsprop_rmsprop_001_eta], linewidth=1, linestyle='-', label='rmsprop-0.001')
    axes.plot(lst_epoch, [np.linalg.norm(xx) for xx in lst_rmsprop_rmsprop_01_eta], linewidth=1, linestyle='-', label='rmsprop-0.01')

    axes.plot(lst_epoch, [np.linalg.norm(xx) for xx in lst_adam_adam_0005_eta], linewidth=1, linestyle='-', label='adam-0.0005')
    axes.plot(lst_epoch, [np.linalg.norm(xx) for xx in lst_adam_adam_001_eta], linewidth=1, linestyle='-', label='adam-0.001')
    axes.plot(lst_epoch, [np.linalg.norm(xx) for xx in lst_adam_adam_01_eta], linewidth=1.5, linestyle='-', label='adam-0.01')

    # axes.plot(lst_epoch, [np.linalg.norm(xx) for xx in lst_amsgrad_amsgrad_0005_eta], linewidth=1, linestyle='-', label='amsgrad-0.0005')
    # axes.plot(lst_epoch, [np.linalg.norm(xx) for xx in lst_amsgrad_amsgrad_001_eta], linewidth=1, linestyle='-', label='amsgrad-0.001')
    axes.plot(lst_epoch, [np.linalg.norm(xx) for xx in lst_amsgrad_amsgrad_01_eta], linewidth=1, linestyle='-', label='amsgrad-0.01')

    axes.plot(lst_epoch, [np.linalg.norm(xx) for xx in lst_gd_newton_eta], linewidth=2, linestyle='-.', label='gdn')
    axes.plot(lst_epoch, [np.linalg.norm(xx) for xx in lst_newton_newton_eta], linewidth=1, linestyle='-.', label='cn')
    axes.set_yscale('log')
    axes.set_xlabel("epoch", fontsize=15)
    axes.set_ylabel(r"generator $\vert| \eta \vert|$", fontsize=15)
    axes.legend(loc='center', bbox_to_anchor=(0.35, 0.4), framealpha=0.5, fontsize=10)

    # # axes[1].plot(lst_epoch, [np.linalg.norm(xx) for xx in lst_rmsprop_rmsprop_0001_w], linewidth=1, linestyle='-', label='rmsprop-0.0001')
    # axes[1].plot(lst_epoch, [np.linalg.norm(xx) for xx in lst_rmsprop_rmsprop_0005_w], linewidth=1, linestyle='-', label='rmsprop-0.0005')
    # axes[1].plot(lst_epoch, [np.linalg.norm(xx) for xx in lst_rmsprop_rmsprop_001_w], linewidth=1, linestyle='-', label='rmsprop-0.001')
    # axes[1].plot(lst_epoch, [np.linalg.norm(xx) for xx in lst_rmsprop_rmsprop_01_w], linewidth=1, linestyle='-', label='rmsprop-0.01')

    # axes[1].plot(lst_epoch, [np.linalg.norm(xx) for xx in lst_adam_adam_0005_w], linewidth=1, linestyle='-', label='adam-0.0005')
    # axes[1].plot(lst_epoch, [np.linalg.norm(xx) for xx in lst_adam_adam_001_w], linewidth=1, linestyle='-', label='adam-0.001')
    # axes[1].plot(lst_epoch, [np.linalg.norm(xx) for xx in lst_adam_adam_01_w], linewidth=1, linestyle='-', label='adam-0.01')

    # axes[1].plot(lst_epoch, [np.linalg.norm(xx) for xx in lst_amsgrad_amsgrad_0005_w], linewidth=1, linestyle='-', label='amsgrad-0.0005')
    # axes[1].plot(lst_epoch, [np.linalg.norm(xx) for xx in lst_amsgrad_amsgrad_001_w], linewidth=1, linestyle='-', label='amsgrad-0.001')
    # axes[1].plot(lst_epoch, [np.linalg.norm(xx) for xx in lst_amsgrad_amsgrad_01_w], linewidth=1, linestyle='-', label='amsgrad-0.01')

    # axes[1].plot(lst_epoch, [np.linalg.norm(xx) for xx in lst_gd_newton_w], linewidth=2, linestyle='-.', label='gdn')
    # axes[1].plot(lst_epoch, [np.linalg.norm(xx) for xx in lst_newton_newton_w], linewidth=1, linestyle='-.', label='cn')
    # axes[1].set_yscale('log')
    # # axes[1].set_xlabel("epoch", fontsize=15)
    # axes[1].set_title(r"discrimiantor $\vert| \omega \vert|$", fontsize=15)
    # axes[1].legend(loc='center', bbox_to_anchor=(0.3, 0.4), fontsize=10)

    # fig.subplots_adjust(hspace=0.0, wspace=0.0)
    # fig.subplots_adjust(left=0.15, right=0.99, top=0.98, bottom=0.15)

    plt.tight_layout()

    plt.show()


def plot_gmm():
    pattern = "./checkpoints/gmm/_{}-{}/{}-epoch_{}.tar"

    def get_lst(g_optim, g_step_size, d_optim, d_step_size, d_num_step):
        # pattern = "./checkpoints/gmm/{}-{}-{}-{}-{}-new_new/{}-epoch_{:d}.tar"
        pattern = "./checkpoints/gmm/{}-{}-{}-{}-{}-new/{}-epoch_{:d}.tar"

        lst_d_discriminator_norm = []
        lst_d_generator_norm = []

        lst_epoch = np.arange(1, 200, 1)

        for i in lst_epoch:
            ckpt = torch.load(pattern.format(g_optim, g_step_size, d_optim, d_step_size, d_num_step, "discriminator", i), map_location='cpu')
            lst_d_discriminator_norm.append(norm(ckpt['gradient']))

            ckpt = torch.load(pattern.format(g_optim, g_step_size, d_optim, d_step_size, d_num_step, "generator", i), map_location='cpu')
            lst_d_generator_norm.append(norm(ckpt['gradient']))


        return lst_epoch, lst_d_discriminator_norm, lst_d_generator_norm

    # lst_epoch, lst_sd_gd_discriminator, lst_sd_gd_generator = get_lst("sd", 0.01, "gd", 0.01, 1)
    # lst_epoch, lst_gd_fr_discriminator, lst_gd_fr_generator = get_lst("gd", 0.01, "fr", 0.01, 1)
    # lst_epoch, lst_gd_newton_discriminator, lst_gd_newton_generator = get_lst("gd", 0.01, "newton", 1.0, 1)
    lst_epoch, lst_newton_newton_discriminator, lst_newton_newton_generator = get_lst("newton", 1.0, "newton", 1.0, 1)

    fig, ax = plt.subplots(figsize=(4, 3), nrows=1, ncols=1)
    ax.plot(lst_epoch, lst_sd_gd_discriminator, label='tgda')
    ax.plot(lst_epoch, lst_gd_fr_discriminator, label='fr')
    ax.plot(lst_epoch, lst_gd_newton_discriminator, label='gdn')
    ax.plot(lst_epoch, lst_newton_newton_discriminator, label='cn')
    ax.set_yscale('log')
    ax.set_ylabel("discriminator grad norm", fontsize=14)
    ax.yaxis.set_label_position("right")
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    plt.legend(fontsize=14)

    fig, ax = plt.subplots(figsize=(4, 3), nrows=1, ncols=1)
    ax.plot(lst_epoch, lst_sd_gd_generator, label='tgda')
    ax.plot(lst_epoch, lst_gd_fr_generator, label='fr')
    ax.plot(lst_epoch, lst_gd_newton_generator, label='gdn')
    ax.plot(lst_epoch, lst_newton_newton_generator, label='cn')
    ax.set_yscale('log')
    ax.set_ylabel("generator grad norm", fontsize=14)
    ax.yaxis.set_label_position("right")
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    plt.legend(fontsize=14)

    plt.show()


    # discriminator = DNet().double()
    # discriminator.load_state_dict(torch.load(pattern.format(g_optim, d_optim, "discriminator", epoch), map_location='cpu')['model_state_dict'])

    # plt.figure(figsize=(4, 3))

    # plot_db_discriminator(lambda xx: discriminator(xx).sigmoid(), -5, 5, -5, 5)
    # plt.tight_layout()

    # generator = GNet().double()
    # generator.load_state_dict(torch.load(pattern.format(g_optim, d_optim, "generator", epoch), map_location='cpu')['model_state_dict'])

    # noise = torch.randn(20000, 16).double()
    # fake_data = generator(noise).detach().numpy()

    # plt.figure(figsize=(4, 3))

    # sns.kdeplot(fake_data[:, 0], fake_data[:, 1], shade='True')


    # i = 10
    # discriminator = torch.load("./checkpoints/discriminator-epoch_{:03d}.pth".format(i), map_location='cpu').eval()
    # generator = torch.load("./checkpoints/generator-epoch_{:03d}.pth".format(i), map_location='cpu').eval()

    # dataset = torch.randn(30, 2)
    # dataset = torch.utils.data.TensorDataset(dataset)

    # loader = torch.utils.data.DataLoader(dataset, batch_size=30, shuffle=True)

    # plot_visualization(discriminator, generator, loader)
    # plt.show()


def plot_mnist(g_optim, d_optim, epoch):
    pattern = "./checkpoints/mnist/{}-{}/{}-epoch_{}.tar"

    discriminator = Discriminator().double()
    discriminator.load_state_dict(torch.load(pattern.format(g_optim, d_optim, "discriminator", epoch), map_location='cpu')['model_state_dict'])

    sigma, _ = power(discriminator.fc2.weight, discriminator.fc2.u, maxiter=10)
    print(sigma)

    generator = Generator().double()
    generator.load_state_dict(torch.load(pattern.format(g_optim, d_optim, "generator", epoch), map_location='cpu')['model_state_dict'])

    noise = torch.randn(25, 100).double()
    fake_data = vectors2images(generator(noise)).detach()

    grid = torchvision.utils.make_grid(vectors2images(fake_data), nrow=5, normalize=True, range=(-1, 1), padding=0)

    fig = plt.figure(figsize=(4, 4))
    grid = grid.numpy()
    plt.imshow(np.transpose(grid, (1, 2, 0)), interpolation='nearest')
    plt.axis('off')

    # fig, axes = plt.subplots(figsize=(10, 2), nrows=1, ncols=5)
    # for i in range(5):
    #     axes[i].imshow(fake_data[i][0], cmap='gray')

    # plt.subplots_adjust(hspace=0.0, wspace=0.0)
    # plt.subplots_adjust(left=0.05, right=0.95, top=1.0, bottom=0.0)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # parser.add_argument("--epoch", type=int, default=199)
    parser.add_argument("--dataset", type=str, default="single_gaussian", help="single_gaussian | single_gaussian_ill_conditioned")

    # parser.add_argument("--d_optim", type=str, default="gd", help="gd | newton")
    # parser.add_argument("--g_optim", type=str, default="gd", help="gd | newton | fr")

    args = parser.parse_args()
    print(args)

    set_seed(0)

    if args.dataset == "single_gaussian":
        plot_single_gaussian()

    elif args.dataset == "single_gaussian_ill_conditioned":
        plot_single_gaussian()

    elif args.dataset == "covariance":
        plot_covariance()

    elif args.dataset == "gmm":
        plot_gmm()

    # elif args.dataset == "mnist":
    #     plot_mnist(args.g_optim, args.d_optim, args.epoch)

    elif args.dataset == "rmsprop_vs_newton":
        plot_rmsprop_vs_newton()
