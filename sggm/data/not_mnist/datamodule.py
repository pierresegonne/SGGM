import multiprocessing
import numpy as np
import pathlib
import pytorch_lightning as pl
import torch

from scipy.io import loadmat
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST

N_cpus = multiprocessing.cpu_count()

DATA_FILENAME = "notMNIST_small.mat"


class NotMNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        n_workers: int,
        train_val_split: float = 0.9,
        test_split: float = 0.1667,
        **kwargs,
    ):
        super().__init__()

        self.batch_size = batch_size

        # Multiprocessing
        self.n_workers = n_workers if n_workers is not None else N_cpus
        self.pin_memory = True if self.n_workers > 0 else False

        # Splits
        self.test_split = test_split
        self.train_val_split = train_val_split

        # Manual
        self.x_mean = 0.1307
        self.x_std = 0.3081
        self.range = 255
        self.dims = (1, 28, 28)
        self.labels = [
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            "j",
        ]

    def setup(self, stage: str = None):
        mat = loadmat(f"{pathlib.Path(__file__).parent.absolute()}/{DATA_FILENAME}")
        X, Y = mat["images"], mat["labels"]
        # Resize X from w x h x N to N x 1 x w x h
        X = np.moveaxis(X, [2], [0])
        X = X[:, None, :, :]
        Y = Y.reshape((-1, 1))

        # Shuffle
        idx_shuffler = np.random.permutation(X.shape[0])
        X = X[idx_shuffler, :]
        Y = Y[idx_shuffler, :]

        # Split train_val and test
        idx_test_split = int(X.shape[0] * self.test_split)
        x_train, x_test = X[idx_test_split:], X[:idx_test_split]
        y_train, y_test = Y[idx_test_split:], Y[:idx_test_split]

        # Standardisation
        x_train /= self.range
        x_test /= self.range
        self.x_mean = np.mean(x_train)
        self.x_std = np.std(x_train)
        x_train = (x_train - self.x_mean) / self.x_std
        x_test = (x_test - self.x_mean) / self.x_std

        # To torch Dataset
        train_size = int(x_train.shape[0] * self.train_val_split)
        val_size = x_train.shape[0] - train_size
        self.train_dataset = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
        self.train_dataset, self.val_dataset = random_split(
            self.train_dataset, [train_size, val_size]
        )

        self.test_dataset = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            pin_memory=self.pin_memory,
        )


if __name__ == "__main__":

    dm = NotMNISTDataModule(256, 0)
    dm.setup()

    # Observe sample
    x, y = next(iter(dm.train_dataloader()))

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.imshow((x[0, 0, :, :] * dm.x_std + dm.x_mean) * dm.range, cmap="binary")
    print(f"Displayed '{dm.labels[int(y[0])]}'")

    plt.show()

    # Info about the data
