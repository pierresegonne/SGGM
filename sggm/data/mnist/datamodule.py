import multiprocessing
import os
import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

N_cpus = multiprocessing.cpu_count()


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        n_workers: int,
        train_val_split: float = 0.9,
    ):
        super().__init__()

        self.batch_size = batch_size

        # Multiprocessing
        self.n_workers = n_workers if n_workers is not None else N_cpus
        self.pin_memory = True if self.n_workers > 0 else False

        self.train_val_split = train_val_split

        # Manual
        self.x_mean = 0.1307
        self.x_std = 0.3081
        self.range = 255
        self.dims = (1, 28, 28)

    def setup(self, stage: str = None):

        # Transforms
        mnist_transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([self.x_mean], [self.x_std])]
        )

        # Train
        train_val = MNIST(
            os.path.dirname(__file__),
            download=True,
            train=True,
            transform=mnist_transforms,
        )
        train_length = int(len(train_val) * self.train_val_split)
        val_length = len(train_val) - train_length
        self.train_dataset, self.val_dataset = random_split(
            train_val, [train_length, val_length]
        )

        # Test
        self.test_dataset = MNIST(
            os.path.dirname(__file__),
            download=True,
            train=False,
            transform=mnist_transforms,
        )

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

    dm = MNISTDataModule(256, 0)
    dm.setup()

    # Observe sample
    x, y = next(iter(dm.train_dataloader()))

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.imshow((x[0, 0, :, :] * dm.x_std + dm.x_mean) * dm.range, cmap="binary")
    print(f"Displayed {y[0]}")

    plt.show()

    # Info about the data
