import multiprocessing
import pytorch_lightning as pl
import torch

from torch.utils.data import TensorDataset, DataLoader
from sggm.data.regression_datamodule import RegressionDataModule
from sggm.data.sanity_check.prior import prior_α, prior_β
from sggm.definitions import SANITY_CHECK_MAX_BATCH_ITERATIONS


class SanityCheckDataModule(RegressionDataModule):
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
        self.testing_range = [-0.5, 10.5]

        # Manual as we know it
        self.dims = 1
        self.out_dims = 1

        self.max_batch_iterations = SANITY_CHECK_MAX_BATCH_ITERATIONS

    def setup(self, stage: str = None):
        # Save mean and std
        self.x_mean = 0
        self.x_std = 1
        self.y_mean = 0
        self.y_std = 1

        x_train = torch.FloatTensor(self.N_train, 1).uniform_(*self.training_range)
        eps = torch.randn_like(x_train)
        y_train = self.data_mean(x_train) + self.data_std(x_train) * eps
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
        return torch.ones_like(x)

    @staticmethod
    def data_std(x: torch.Tensor) -> torch.Tensor:
        # sig = 4.47213595499958
        # lambda = 0.05
        return torch.sqrt(
            torch.Tensor([prior_β]) / (torch.Tensor([prior_α]) - 1)
        ) * torch.ones_like(x)


if __name__ == "__main__":

    dm = SanityCheckDataModule(1000, 0)
    dm.setup()

    x_train, y_train = next(iter(dm.train_dataloader()))

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.scatter(x_train, y_train)

    plt.show()
