import math
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rc('pdf', fonttype=42)


def initialize():
    # closer to local minimax
    return 1, math.pi / 2 + 1e-2

    # closer to local minimum
    # return 1, - math.pi / 2 + 1e-2


def f(x, y):
    return (x ** 2 + 1) * (2 + np.sin(y))


def dx(x, y):
    return 2 * x * (2 + np.sin(y))


def dy(x, y):
    return (x ** 2 + 1) * np.cos(y)


def hxx(x, y):
    return 2 * (2 + np.sin(y))


def hxy(x, y):
    return 2 * x * np.cos(y)


def hyy(x, y):
    return - (x ** 2 + 1) * np.sin(y)


def hyx(x, y):
    return 2 * x * np.cos(y)


def get_x_update(x, y, x_optim, x_step_size, y_optim, y_step_size):
    if x_optim == "gd":
        return - x_step_size * dx(x, y)

    elif x_optim == "sd":
        return - x_step_size * dx(x, y) + x_step_size * hxy(x, y) * 1 / hyy(x, y) * dy(x, y)

    elif x_optim == "newton":
        return - x_step_size / (hxx(x, y) - hxy(x, y) * 1 / hyy(x, y) * hyx(x, y)) * dx(x, y)


def get_y_update(x, y, x_optim, x_step_size, y_optim, y_step_size):
    if y_optim == "gd":
        return y_step_size * dy(x, y)

    elif y_optim == "fr":
        return y_step_size * dy(x, y) + x_step_size * 1 / hyy(x, y) * hyx(x, y) * dx(x, y)

    elif y_optim == "newton":
        return - y_step_size / hyy(x, y) * dy(x, y)


def train(epoch=100, x_optim="gd", x_step_size=0.01, y_optim="gd", y_step_size=0.01, simultaneous=False):
    x, y = initialize()

    lst_xy = []

    for i in range(epoch):
        if simultaneous:
            x_update = get_x_update(x, y, x_optim, x_step_size, y_optim, y_step_size)
            y_update = get_y_update(x, y, x_optim, x_step_size, y_optim, y_step_size)

            x = x + x_update
            y = y + y_update

        else:
            x_update = get_x_update(x, y, x_optim, x_step_size, y_optim, y_step_size)
            x = x + x_update

            y_update = get_y_update(x, y, x_optim, x_step_size, y_optim, y_step_size)
            y = y + y_update

        print("epoch: {:4d}, x: {:.12f}, y: {:.12f}".format(i, x, y))
        lst_xy.append((x, y))

    return lst_xy


def plot():
    lst_gd_gd = train(200, x_optim="gd", x_step_size=0.1, y_optim="gd", y_step_size=0.1, simultaneous=False)
    lst_sd_gd = train(200, x_optim="sd", x_step_size=0.1, y_optim="gd", y_step_size=0.1, simultaneous=True)
    lst_gd_fr = train(200, x_optim="gd", x_step_size=0.1, y_optim="fr", y_step_size=0.1, simultaneous=True)
    lst_gd_newton = train(200, x_optim="gd", x_step_size=0.1, y_optim="newton", y_step_size=1, simultaneous=False)
    lst_newton_newton = train(200, x_optim="newton", x_step_size=1, y_optim="newton", y_step_size=1, simultaneous=False)


    def dist2optimal(xx, yy):
        return np.sqrt(xx ** 2 + (yy - np.pi / 2) ** 2)

    plt.figure(figsize=(3, 3))
    ax = plt.axes()
    ax.plot([dist2optimal(xx, yy) for xx, yy in lst_gd_gd], linewidth=1, linestyle=':', label='gda', color='tab:red')
    ax.plot([dist2optimal(xx, yy) for xx, yy in lst_sd_gd], linewidth=1, linestyle='-.', label='tgda', color='tab:olive')
    ax.plot([dist2optimal(xx, yy) for xx, yy in lst_gd_fr], linewidth=1, linestyle='-.', label='fr', color='tab:pink')
    ax.plot([dist2optimal(xx, yy) for xx, yy in lst_gd_newton], linewidth=1, linestyle='--', label='gdn', color='tab:blue')
    ax.plot([dist2optimal(xx, yy) for xx, yy in lst_newton_newton], linewidth=1, linestyle='-', label='cn', color='tab:orange')
    ax.legend(loc='center', bbox_to_anchor=(0.25, 0.3), fontsize=10)

    ax.set_yscale('log')
    ax.set_xlabel("epoch")
    # ax.legend(loc='lower left')

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    plot()

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--epoch", type=int, default=10)

    parser.add_argument("--x_optim", type=str, default="gd", help="gd | newton")
    parser.add_argument("--x_step_size", type=float, default=0.01)

    parser.add_argument("--y_optim", type=str, default="gd", help="gd | newton | fr")
    parser.add_argument("--y_step_size", type=float, default=0.01)

    parser.add_argument("--simultaneous", type=int, default=0)

    args = parser.parse_args()
    print(args)

    train(epoch=args.epoch,
          x_optim=args.x_optim, x_step_size=args.x_step_size,
          y_optim=args.y_optim, y_step_size=args.y_step_size,
          simultaneous=args.simultaneous,
          )

    """Plot the landscape"""
    x = np.linspace(- 1, 1, 100)
    y = np.linspace(- math.pi, math.pi, 100)

    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    # ax.scatter(0, math.pi / 2, f(0, math.pi / 2), marker='*', markersize=20)
    ax.set_xlabel(r'x')
    ax.set_ylabel(r'y')
    plt.tight_layout()
    plt.show()
