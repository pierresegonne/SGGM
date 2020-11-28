import torch

from torch.utils.data import TensorDataset, DataLoader

from sggm.data.regression_datamodule import RegressionDataModule
from sggm.definitions import TOY_MAX_BATCH_ITERATIONS


class ToyDataModule(RegressionDataModule):
    def __init__(
        self,
        batch_size: int,
        n_workers: int,
        N_train: int = 625,
        N_test: int = 1000,
        train_val_split: float = 0.9,
        **kwargs,
    ):
        super(ToyDataModule, self).__init__(
            batch_size,
            n_workers,
            train_val_split,
            **kwargs,
        )
        self.N_train = N_train
        self.N_test = N_test

        self.training_range = [0, 10]
        self.testing_range = [-2.5, 12.5]

        # Manual as we know it
        self.dims = 1
        self.out_dims = 1

        self.max_batch_iterations = TOY_MAX_BATCH_ITERATIONS

    def setup(self, stage: str = None):
        # Save mean and std
        self.x_mean = 0
        self.x_std = 1
        self.y_mean = 0
        self.y_std = 1

        x_train = torch.FloatTensor(self.N_train, 1).uniform_(*self.training_range)
        eps1, eps2 = torch.randn_like(x_train), torch.randn_like(x_train)
        y_train = self.data_mean(x_train) + 0.3 * eps1 + 0.3 * x_train * eps2
        x_test = torch.FloatTensor(self.N_test, 1).uniform_(*self.testing_range)
        y_test = self.data_mean(x_test)

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
    def data_mean(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sin(x)

    @staticmethod
    def data_std(x: torch.Tensor) -> torch.Tensor:
        return torch.abs(0.3 * torch.sqrt(1 + x * x))
