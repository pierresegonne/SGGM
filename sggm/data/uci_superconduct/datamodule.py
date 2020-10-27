import multiprocessing
import pandas as pd
import pathlib
import pytorch_lightning as pl
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

N_cpus = multiprocessing.cpu_count()

DATA_FILENAME = "raw.csv"
"""
Link to get the raw.csv data: https://archive.ics.uci.edu/ml/datasets/Superconductivty+Data
"""


class UCISuperConductDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        n_workers: int,
        train_val_split: float = 0.8,
        test_split: float = 0.1,
        **kwargs,
    ):
        super(UCISuperConductDataModule, self).__init__()
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.test_split = test_split

        self.n_workers = n_workers if n_workers is not None else N_cpus
        self.pin_memory = True if self.n_workers > 0 else False

        # Manual as we know it
        self.dims = 81
        self.out_dims = 1

    def setup(self, stage: str = None):

        df = pd.read_csv(f"{pathlib.Path(__file__).parent.absolute()}/{DATA_FILENAME}")
        # Split features, targets
        x = df.drop(columns=["critical_temp"]).values
        y = df["critical_temp"].values

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
