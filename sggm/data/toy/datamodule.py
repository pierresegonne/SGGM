import torch

from torch.utils.data import TensorDataset, DataLoader

from sggm.data.regression_datamodule import RegressionDataModule
from sggm.data.shifted import DataModuleShifted
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
        RegressionDataModule.__init__(
            self,
            batch_size,
            n_workers,
            train_val_split,
        )
        self.N_train = N_train
        self.N_test = N_test

        self.training_range = [0, 10]
        self.testing_range = [-3.5, 12.5]

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
        self.setup_train_val_datasets(self.train_dataset)
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


class ToyDataModuleShifted(ToyDataModule, DataModuleShifted):
    def __init__(
        self,
        batch_size: int,
        n_workers: int,
        N_train: int = 625,
        N_test: int = 1000,
        train_val_split: float = 0.9,
        shifting_proportion_total: float = 0.1,
        shifting_proportion_k: float = 1e-2,
        **kwargs,
    ):
        ToyDataModule.__init__(
            self,
            batch_size,
            n_workers,
            N_train,
            N_test,
            train_val_split,
        )
        DataModuleShifted.__init__(
            self, shifting_proportion_total, shifting_proportion_k
        )

    def setup(self, stage: str = None):
        ToyDataModule.setup(self, stage)
        DataModuleShifted.setup(self)
