import multiprocessing
import os
import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision import transforms
from torchvision.datasets import FashionMNIST

from sggm.types_ import List

N_cpus = multiprocessing.cpu_count()


class FashionMNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        n_workers: int,
        train_val_split: float = 0.9,
        **kwargs,
    ):
        super().__init__()

        self.batch_size = batch_size

        # Multiprocessing
        self.n_workers = n_workers if n_workers is not None else N_cpus
        self.pin_memory = True if self.n_workers > 0 else False

        self.train_val_split = train_val_split

        # Manual
        self.x_mean = 0.2860
        self.x_std = 0.3530
        self.range = 255
        self.x_mean = 0.0
        self.x_std = 1
        self.dims = (1, 28, 28)
        self.labels = [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ]

    def setup(self, stage: str = None):

        # Transforms
        mnist_transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([self.x_mean], [self.x_std])]
        )

        # Train
        train_val = FashionMNIST(
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
        self.test_dataset = FashionMNIST(
            os.path.dirname(__file__),
            download=True,
            train=False,
            transform=mnist_transforms,
        )

    def train_dataloader(self) -> DataLoader:
        # Put on GPU
        # Eugene's hack
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        values = torch.cat([e[0] for e in self.train_dataset], dim=0)
        targets = torch.tensor([e[1] for e in self.train_dataset])
        self.train_dataset = TensorDataset(values.to(device), targets.to(device))
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


class FashionMNISTDataModuleND(FashionMNISTDataModule):
    def __init__(
        self,
        batch_size: int,
        n_workers: int,
        train_val_split: float = 0.9,
        # To keep notation consistent, even though that represent the idx of classes
        digits: List[int] = [0, 1],
        **kwargs,
    ):
        super().__init__(batch_size, n_workers, train_val_split, **kwargs)
        assert len(digits) >= 2
        assert len(digits) == len(set(digits))
        for d in digits:
            assert isinstance(d, int)
            assert (0 <= d) & (d <= 9)
        self.digits = digits

    def setup(self, stage: str = None):
        # Transforms
        mnist_transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([self.x_mean], [self.x_std])]
        )

        # Train
        train_val = FashionMNIST(
            os.path.dirname(__file__),
            download=True,
            train=True,
            transform=mnist_transforms,
        )

        idx = torch.cat(
            [train_val.targets[:, None] == digit for digit in self.digits], dim=1
        ).any(dim=1)
        train_val.targets = train_val.targets[idx]
        train_val.data = train_val.data[idx]

        train_length = int(len(train_val) * self.train_val_split)
        val_length = len(train_val) - train_length
        self.train_dataset, self.val_dataset = random_split(
            train_val, [train_length, val_length]
        )

        # Test
        self.test_dataset = FashionMNIST(
            os.path.dirname(__file__),
            download=True,
            train=False,
            transform=mnist_transforms,
        )
        idx = torch.cat(
            [self.test_dataset.targets[:, None] == digit for digit in self.digits],
            dim=1,
        ).any(dim=1)
        self.test_dataset.targets = self.test_dataset.targets[idx]
        self.test_dataset.data = self.test_dataset.data[idx]


if __name__ == "__main__":

    # dm = FashionMNISTDataModule(256, 0)
    dm = FashionMNISTDataModuleND(16, 0, digits=[2, 4, 5])
    dm.setup()

    # Observe sample
    x, y = next(iter(dm.train_dataloader()))

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.imshow((x[0, 0, :, :] * dm.x_std + dm.x_mean) * dm.range, cmap="binary")
    print(f"Displayed '{dm.labels[y[0]]}'")

    plt.show()

    # Info about the data
