import torch
from torch import nn
import torch.nn.functional as F

from utils import *


def _l2normalize(v, eps=1e-12):
    return v / (torch.norm(v) + eps)


def power(W, u, maxiter=1):
    with torch.no_grad():
        for i in range(maxiter):
            v = _l2normalize(W.t().mm(u))
            u = _l2normalize(W.mm(v))

    sigma = torch.sum(u * W.mm(v))
    return sigma, u.clone().detach()


class SNLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('u', torch.randn(out_features, 1))

    @property
    def W_(self):
        sigma, self.u = power(self.weight, self.u)
        return self.weight / sigma

    def forward(self, x):
        return F.linear(x, self.W_, self.bias)


class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = SNLinear(784, 512)
        self.fc2 = SNLinear(512, 512)
        self.fc3 = SNLinear(512, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
        x = self.fc3(x)
        return x


class Generator(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 784)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
        x = self.fc3(x).tanh()
        return x


class DNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.ac = F.relu
        # self.ac = F.softplus
        self.fc1 = nn.Linear(2, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.ac(self.fc1(x))
        x = self.ac(self.fc2(x))
        x = self.ac(self.fc3(x))
        x = self.fc4(x)
        return x


class GNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.ac = F.relu
        # self.ac = F.softplus
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.ac(self.fc1(x))
        x = self.ac(self.fc2(x))
        x = self.ac(self.fc3(x))
        x = self.fc4(x)
        return x


class OneLayerNet(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        # make sure the initialization is close to zero
        # otherwise, newton step might overshoot in some cases
        self.w = nn.Parameter(1e-1 * torch.randn((input_dim, 1), dtype=torch.double))

    def forward(self, x):
        return x.mm(self.w)


class ShiftNet(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        # make sure the initialization is close to zero
        # otherwise, newton step might overshoot in some cases
        self.eta = nn.Parameter(1e-1 * torch.randn((input_dim, ), dtype=torch.float))

    def forward(self, x):
        return x + self.eta

    def get_numpy_eta(self):
        return self.eta.detach().cpu().numpy()


class QuadraticNet(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.W = nn.Parameter(1e-1 * torch.randn((input_dim, input_dim), dtype=torch.double))

    def forward(self, x):
        return (x.mm(self.W) * x).sum(dim=1, keepdim=True)


class AffineNet(nn.Module):

    def __init__(self, input_dim=2, output_dim=2):
        super().__init__()
        self.V = nn.Parameter(torch.tensor([[1., 0.], [0., 0.2]]) + 1e-2 * torch.randn((input_dim, output_dim)))

    def forward(self, z):
        """Return V z"""
        return z.mm(self.V)

    def get_numpy_eta(self):
        return self.V.detach().cpu().numpy()


if __name__ == "__main__":
    W = torch.tensor([[2, 0, 0], [0, 3.5, 0]], dtype=torch.float)
    U, _ = torch.qr(torch.randn(2, 2, dtype=torch.float))
    V, _ = torch.qr(torch.randn(3, 3, dtype=torch.float))

    W = U.mm(W).mm(V.t())

    u = torch.randn(2, 1, dtype=torch.float)
    sigma, _u = power(W, u, maxiter=2)
    print(sigma)
