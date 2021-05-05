import torch

from numpy import pi
from torch.utils.data import TensorDataset

"""
Helper for shifted datamodules
"""


def t_log(x: float) -> torch.Tensor:
    return torch.log(torch.Tensor([x]))


def log_density(x: torch.Tensor) -> torch.Tensor:
    """Computes the density of points for a set of points

    Args:
        x (torch.Tensor): set of points (N x D)

    Returns:
        torch.Tensor: density in # points per unit of volume in the set
    """
    hypercube_min, _ = torch.min(x, dim=0)
    hypercube_max, _ = torch.max(x, dim=0)

    log_vol = torch.sum(torch.log(hypercube_max - hypercube_min), 0)
    return torch.log(torch.Tensor([x.shape[0]])) - log_vol


def log_radius(p_tot: float, p_k: float, x: torch.Tensor) -> torch.Tensor:
    """Computes the radius of the holes to introduce in the training data

    Args:
        p_tot (float): proportion of total expected points in B_1 U B_2 U ... B_K
        p_k (float): proportion of points sampled as {B_k}_{k=1}^{K} centers
        x (torch.Tensor): sets of points

    Returns:
        torch.Tensor: radius of any B_k
    """
    log_d = log_density(x)
    D = x.shape[1]

    # log_r = torch.pow(
    #     (p_tot / p_k)
    #     * (torch.lgamma(torch.Tensor([D / 2 + 1])) - log_d).exp()
    #     / (pi ** (D / 2)),
    #     1 / D,
    # )
    log_r = (
        1
        / D
        * (
            t_log(p_tot / p_k)
            + torch.lgamma(torch.Tensor([D / 2 + 1]))
            - log_d
            - t_log(pi ** (D / 2))
        )
    )
    return log_r


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
    idx_k = torch.multinomial(
        torch.ones_like(x_train[:, 0]).flatten(), K, replacement=False
    )
    x_k = x_train[idx_k]

    # Determine average distance between points
    log_dist = log_radius(shifting_proportion_total, shifting_proportion_k, x_train)

    # Any point laying inside any hyperball gets affected to test
    in_any_b_k = torch.where(
        torch.where(torch.log(torch.cdist(x_train, x_k)) < log_dist, 1, 0).sum(dim=1)
        >= 1,
        1,
        0,
    )

    x_test = torch.cat((x_test, x_train[in_any_b_k == 1]), dim=0)
    y_test = torch.cat((y_test, y_train[in_any_b_k == 1]), dim=0)

    x_train = x_train[in_any_b_k == 0]
    y_train = y_train[in_any_b_k == 0]

    return (x_train, y_train), (x_test, y_test)


class DataModuleShifted:
    """
    Add-on class to introduce shift between the training and testing distibutions.
    [NOTE]: Implicitely assumes that is used in combinaison with a children of another DataModule
    """

    def __init__(
        self,
        shifting_proportion_total: float = 0.1,
        shifting_proportion_k: float = 1e-2,
        *args,
        **kwargs
    ):
        self.shifting_proportion_total = float(shifting_proportion_total)
        self.shifting_proportion_k = float(shifting_proportion_k)

    def setup(self):
        train, test = generate_shift(
            (self.shifting_proportion_total, self.shifting_proportion_k),
            self.train_dataset.dataset.tensors,
            self.test_dataset.tensors,
        )

        self.train_dataset = TensorDataset(*train)
        self.setup_train_val_datasets(self.train_dataset)
        self.test_dataset = TensorDataset(*test)


class DataModuleShiftedSplit:
    """
    Add-on class to introduce shift based on splits between the training and testing distibutions.
    Based on: https://arxiv.org/abs/1906.11537 ['In-Between'] Uncertainty in BNN
    [NOTE]: Implicitely assumes that is used in combinaison with a children of another DataModule
    """

    def setup(self, dim_idx: int):

        x_train, y_train = self.train_dataset.dataset.tensors
        x_test, y_test = self.test_dataset.tensors

        _, dim_col_indices = x_train[:, dim_idx].sort()
        N = dim_col_indices.shape[0]
        split_indices = dim_col_indices[int(N / 3) : int((2 * N) / 3)]
        non_split_indices = torch.cat(
            (dim_col_indices[: int(N / 3)], dim_col_indices[int((2 * N) / 3) :])
        )
        x_test = torch.cat((x_test, x_train[split_indices]), dim=0)
        y_test = torch.cat((y_test, y_train[split_indices]), dim=0)
        x_train, y_train = x_train[non_split_indices], y_train[non_split_indices]

        self.train_dataset = TensorDataset(x_train, y_train)
        self.setup_train_val_datasets(self.train_dataset)
        self.test_dataset = TensorDataset(x_test, y_test)
