import math

import numpy as np

import torch

import torchvision
import torchvision.transforms as transforms


def gmm_2d(num_mix=8, samples=100000):
    ths = np.linspace(0, 2 * np.pi * (num_mix - 1) / num_mix, num_mix)
    xs, ys = 2 * np.cos(ths), 2 * np.sin(ths)

    K = np.random.randint(num_mix, size=samples)
    X = np.zeros(samples)
    Y = np.zeros(samples)

    for i in range(samples):
        cx, cy = xs[K[i]], ys[K[i]]
        X[i], Y[i] = cx + np.random.randn() / 10, cy + np.random.randn() / 10

    Z = np.stack((X, Y), axis=-1)
    return Z


def gmm_1d(num_mix=3, samples=5000):
    ths = np.linspace(-4, 4, 3)
    K = np.random.randint(num_mix, size=samples)
    X = np.zeros(samples)
    for i in range(samples):
        cx = ths[K[i]]
        X[i] = cx + np.random.randn() / 10
    return X


class NoiseGenerator(object):

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.noise = None

    def __call__(self, real_data, generator):
        if self.noise is not None:
            noise = self.noise

        elif generator.__class__.__name__ == "ShiftNet":
            noise = real_data.clone().detach()

        elif generator.__class__.__name__ == "AffineNet":
            noise = real_data.clone().detach()
            noise = noise.mm(torch.tensor([[1., 0.], [0., 5.]], dtype=torch.double, device=self.device))

        elif generator.__class__.__name__ == "GNet":
            # noise = torch.randn(real_data.size(0), 16).double().to(self.device)
            noise = torch.randn(real_data.size(0), 100).double().to(self.device)
            self.noise = noise

        else:
            noise = torch.randn(real_data.size(0), 100).double().to(self.device)
            self.noise = noise

        return noise


def get_data(option, train_size):
    if option == "single_gaussian":
        dataset = torch.randn(train_size, 2).double()
        dataset = torch.utils.data.TensorDataset(dataset)

    elif option == "single_gaussian_ill_conditioned":
        dataset = torch.randn(train_size, 2).double().mm(torch.tensor([[1., 0], [0, math.sqrt(0.05)]], dtype=torch.double))
        dataset = torch.utils.data.TensorDataset(dataset)

    elif option == "covariance":
        dataset = torch.randn(train_size, 2).double().mm(torch.tensor([[1., 0], [0, math.sqrt(0.04)]], dtype=torch.double))
        dataset = torch.utils.data.TensorDataset(dataset)

    elif option == "gmm":
        dataset = torch.from_numpy(gmm_2d(num_mix=8, samples=train_size)).double()
        dataset = torch.utils.data.TensorDataset(dataset)

        """The following is a mixture of four Gaussians (deprecated)"""
        # gaussian1 = torch.tensor([-1., 1.], dtype=torch.double) + 0.1 * torch.randn(train_size, 2, dtype=torch.double)
        # gaussian2 = torch.tensor([1., 1.], dtype=torch.double) + 0.1 * torch.randn(train_size, 2, dtype=torch.double)
        # gaussian3 = torch.tensor([-1., -1.], dtype=torch.double) + 0.1 * torch.randn(train_size, 2, dtype=torch.double)
        # gaussian4 = torch.tensor([1., -1.], dtype=torch.double) + 0.1 * torch.randn(train_size, 2, dtype=torch.double)

        # idx = torch.randint(1, 5, (train_size, 1))

        # dataset = (idx == 1) * gaussian1 + \
        #           (idx == 2) * gaussian2 + \
        #           (idx == 3) * gaussian3 + \
        #           (idx == 4) * gaussian4
        # dataset = torch.utils.data.TensorDataset(dataset)

    elif option == "mnist":
        def preprocess(sample):
            return sample.view((784,)).double() * 2 - 1

        dataset = torchvision.datasets.MNIST('./data', train=True,
                                             transform=transforms.Compose([
                                                 transforms.ToTensor(),
                                                 preprocess]),
                                             download=True)

        idx = (dataset.targets == 1) | (dataset.targets == 0)
        dataset.targets = dataset.targets[idx]
        dataset.data = dataset.data[idx]

    return dataset


if __name__ == "__main__":
    pass
