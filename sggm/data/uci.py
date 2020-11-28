import multiprocessing
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from sggm.data.regression_datamodule import RegressionDataModule
from sggm.definitions import (
    UCI_LARGE_MAX_BATCH_ITERATIONS,
    UCI_SMALL_MAX_BATCH_ITERATIONS,
)

"""
Abstraction level for all UCI regression datamodules
"""

N_cpus = multiprocessing.cpu_count()


class UCIDataModule(RegressionDataModule):
    def __init__(
        self,
        batch_size: int,
        n_workers: int,
        train_val_split: float = 0.9,
        test_split: float = 0.1,
        **kwargs,
    ):
        super(UCIDataModule, self).__init__(
            batch_size,
            n_workers,
            train_val_split,
            test_split,
            **kwargs,
        )

    def setup(self, x: np.ndarray, y: np.ndarray):

        # First split test away
        test_size = int(x.shape[0] * self.test_split)
        x, x_test, y, y_test = train_test_split(x, y, test_size=test_size)
        y, y_test = y[:, None], y_test[:, None]

        # Standardise
        self.x_mean = x.mean(axis=0)[None, :]
        self.x_std = x.std(axis=0)[None, :]
        self.y_mean = y.mean(axis=0)[:, None]
        self.y_std = y.std(axis=0)[:, None]
        x = (x - self.x_mean) / self.x_std
        y = (y - self.y_mean) / self.y_std
        x_test = (x_test - self.x_mean) / self.x_std
        y_test = (y_test - self.y_mean) / self.y_std

        # Register to tensor and generate datasets
        x_train, y_train = torch.FloatTensor(x), torch.FloatTensor(y)
        x_test, y_test = torch.FloatTensor(x_test), torch.FloatTensor(y_test)

        self.train_dataset = TensorDataset(x_train, y_train)
        train_size = int(x_train.shape[0] * self.train_val_split)
        val_size = x_train.shape[0] - train_size
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

    @property
    def max_batch_iterations(self):
        self.check_setup()
        is_large_uci = (
            int(
                len(self.train_dataset) / (self.train_val_split * (1 - self.test_split))
            )
            > 9000
        )
        if is_large_uci:
            return UCI_LARGE_MAX_BATCH_ITERATIONS
        else:
            return UCI_SMALL_MAX_BATCH_ITERATIONS

    @max_batch_iterations.setter
    def max_batch_iterations(self, value):
        """ allows override of parent attribute with children property """
        pass
