import torch

from numpy import pi
from torch.utils.data import TensorDataset

"""
Helper for shifted datamodules
"""


def density(x: torch.Tensor) -> torch.Tensor:
    """Computes the density of points for a set of points

    Args:
        x (torch.Tensor): set of points (N x D)

    Returns:
        torch.Tensor: density in # points per unit of volume in the set
    """
    hypercube_min, _ = torch.min(x, dim=0)
    hypercube_max, _ = torch.max(x, dim=0)

    vol = torch.prod(hypercube_max - hypercube_min, 0)
    return x.shape[0] / vol


def radius(p_tot: float, p_k: float, x: torch.Tensor) -> torch.Tensor:
    """Computes the radius of the holes to introduce in the training data

    Args:
        p_tot (float): proportion of total expected points in B_1 U B_2 U ... B_K
        p_k (float): proportion of points sampled as {B_k}_{k=1}^{K} centers
        x (torch.Tensor): sets of points

    Returns:
        torch.Tensor: radius of any B_k
    """
    d = density(x)
    D = x.shape[1]

    r = torch.pow(
        (p_tot / p_k)
        * torch.lgamma(torch.Tensor([D / 2 + 1])).exp()
        / (d * (pi ** (D / 2))),
        1 / D,
    )
    return r


def generate_shift(
    proportions: tuple([float]),
    train: tuple([torch.Tensor]),
    test: tuple([torch.Tensor]),
) -> tuple([tuple([torch.Tensor])]):
    # Unpack
    shifting_proportion_total, shifting_proportion_k = proportions
    x_train, y_train = train
    x_test, y_test = test

    # Sample p_k% of training samples to serve as center for hyperballs
    K = max(int(shifting_proportion_k * x_train.shape[0]), 1)
    print(K)
    exit()
    idx_k = torch.multinomial(torch.ones_like(x_train).flatten(), K, replacement=False)
    x_k = x_train[idx_k]

    # Determine average distance between points
    dist = radius(shifting_proportion_total, shifting_proportion_k, x_train)
    print("dist", dist)

    # Any point laying inside any hyperball gets affected to test
    in_any_b_k = torch.where(
        torch.where(torch.cdist(x_train, x_k) < dist, 1, 0).sum(dim=1) >= 1,
        1,
        0,
    )

    x_test = torch.cat((x_test, x_train[in_any_b_k == 1]), dim=0)
    y_test = torch.cat((y_test, y_train[in_any_b_k == 1]), dim=0)

    x_train = x_train[in_any_b_k == 0]
    y_train = y_train[in_any_b_k == 0]

    return (x_train, y_train), (x_test, y_test)
