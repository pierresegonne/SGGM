import numpy as np
import torch

from torch.distributions import Normal
from torch.utils.data import TensorDataset, DataLoader
from typing import List

from sggm.data.regression_datamodule import RegressionDataModule
from sggm.definitions import TOY_2D_MAX_BATCH_ITERATIONS

pi = torch.tensor([np.pi])


class Toy2DDataModule(RegressionDataModule):
    def __init__(
        self,
        batch_size: int,
        n_workers: int,
        N_train: int = 2000,
        N_test: int = 1000,
        train_val_split: float = 0.9,
        **kwargs,
    ):
        super(Toy2DDataModule, self).__init__(
            batch_size,
            n_workers,
            train_val_split,
            **kwargs,
        )
        self.N_train = N_train
        self.N_test = N_test

        self.training_range = [2.5, 12.5]
        self.testing_range = [0, 15]

        # Manual as we know it
        self.dims = 2
        self.out_dims = 1

        self.max_batch_iterations = TOY_2D_MAX_BATCH_ITERATIONS

    def setup(self, stage: str = None):
        # Save mean and std
        self.x_mean = 0
        self.x_std = 1
        self.y_mean = 0
        self.y_std = 1

        x_train = self.random_polar(self.N_train, self.training_range)
        eps = torch.randn((self.N_train, 1))
        r = torch.norm(x_train, dim=1, keepdim=True)
        y_train = self.data_mean(r) + self.data_std(r) * eps

        x_test = self.random_polar(self.N_test, self.testing_range)
        r = torch.norm(x_test, dim=1, keepdim=True)
        y_test = self.data_mean(r)

        self.train_dataset = TensorDataset(x_train, y_train)
        train_size = int(self.N_train * self.train_val_split)
        val_size = self.N_train - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.train_dataset, [train_size, val_size]
        )
        self.test_dataset = TensorDataset(x_test, y_test)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            pin_memory=self.pin_memory,
        )

    @staticmethod
    def random_polar(N: int, _range: List[float]) -> torch.Tensor:
        r = torch.rand((N, 1)) * (_range[1] - _range[0]) + _range[0]
        theta = torch.rand((N, 1)) * 2 * pi

        x, y = r * torch.cos(theta), r * torch.sin(theta)
        return torch.cat((x, y), 1)

    @staticmethod
    def data_mean(r: torch.Tensor) -> torch.Tensor:
        return r * torch.sin(r)

    @staticmethod
    def data_std(r: torch.Tensor, with_central_std: bool = False) -> torch.Tensor:
        if with_central_std:
            central_std = 6 * torch.exp(Normal(loc=0, scale=0.5).log_prob(r))
        else:
            central_std = 0
        inc_sin_std = torch.abs(0.3 * torch.sqrt(1 + r * r))
        return central_std + inc_sin_std
