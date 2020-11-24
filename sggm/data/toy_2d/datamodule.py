import multiprocessing
import numpy as np
import pytorch_lightning as pl
import torch

from torch.distributions import Normal
from torch.utils.data import TensorDataset, DataLoader
from typing import List

N_cpus = multiprocessing.cpu_count()
pi = torch.tensor([np.pi])


class Toy2DDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        n_workers: int,
        N_train: int = 2000,
        N_test: int = 1000,
        train_val_split: float = 0.8,
        **kwargs,
    ):
        super(Toy2DDataModule, self).__init__()
        self.batch_size = batch_size
        self.N_train = N_train
        self.N_test = N_test
        self.train_val_split = train_val_split

        self.training_range = [2, 12]
        self.testing_range = [0, 16.5]

        self.n_workers = n_workers if n_workers is not None else N_cpus
        self.pin_memory = True if self.n_workers > 0 else False

        # Manual as we know it
        self.dims = 2
        self.out_dims = 1

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

        self.toy_train = TensorDataset(x_train, y_train)
        train_size = int(self.N_train * self.train_val_split)
        val_size = self.N_train - train_size
        self.toy_train, self.toy_val = torch.utils.data.random_split(
            self.toy_train, [train_size, val_size]
        )
        self.toy_test = TensorDataset(x_test, y_test)

    def train_dataloader(self):
        return DataLoader(
            self.toy_train,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.toy_val,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.toy_test,
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
