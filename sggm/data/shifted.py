import torch
from numpy import pi

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
    d = density(x)
    D = x.shape[1]

    r = torch.pow(
        (p_tot / p_k)
        * torch.lgamma(torch.Tensor([D / 2 + 1])).exp()
        / (d * (pi ** (D / 2))),
        1 / D,
    )
    return r
