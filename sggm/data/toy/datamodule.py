import multiprocessing
import pytorch_lightning as pl
import torch

from torch.utils.data import TensorDataset, DataLoader

N_cpus = multiprocessing.cpu_count()


class ToyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        n_workers: int,
        N_train: int = 625,
        N_test: int = 1000,
        train_val_split: float = 0.8,
        **kwargs,
    ):
        super(ToyDataModule, self).__init__()
        self.batch_size = batch_size
        self.N_train = N_train
        self.N_test = N_test
        self.train_val_split = train_val_split

        self.training_range = [0, 10]
        self.testing_range = [-2.5, 12.5]

        self.n_workers = n_workers if n_workers is not None else N_cpus
        self.pin_memory = True if self.n_workers > 0 else False

        # Manual as we know it
        self.dims = 1
        self.out_dims = 1

    def setup(self, stage: str = None):
        x_train = torch.FloatTensor(self.N_train, 1).uniform_(*self.training_range)
        eps1, eps2 = torch.randn_like(x_train), torch.randn_like(x_train)
        y_train = self.data_mean(x_train) + 0.3 * eps1 + 0.3 * x_train * eps2
        x_test = torch.FloatTensor(self.N_test, 1).uniform_(*self.testing_range)
        y_test = self.data_mean(x_test)

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
    def data_mean(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sin(x)

    @staticmethod
    def data_std(x: torch.Tensor) -> torch.Tensor:
        return torch.abs(0.3 * torch.sqrt(1 + x * x))
