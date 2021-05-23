import multiprocessing
import os
import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision import transforms
from torchvision.datasets import SVHN

N_cpus = multiprocessing.cpu_count()


class SVHNDataModule(pl.LightningDataModule):
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

        # %
        self.train_val_split = train_val_split

        # Manual
        self.x_mean = torch.Tensor([0.4377, 0.4438, 0.4728])
        self.x_std = torch.Tensor([0.1980, 0.2010, 0.1970])
        self.x_mean = 0
        self.x_std = 1
        self.dims = (3, 32, 32)

    def setup(self, stage: str = None):

        # Transforms
        mnist_transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([self.x_mean], [self.x_std])]
        )

        # Train
        train_val = SVHN(
            os.path.dirname(__file__),
            download=True,
            split="train",
            transform=mnist_transforms,
        )
        train_length = int(len(train_val) * self.train_val_split)
        val_length = len(train_val) - train_length
        self.train_dataset, self.val_dataset = random_split(
            train_val, [train_length, val_length]
        )

        # Test
        self.test_dataset = SVHN(
            os.path.dirname(__file__),
            download=True,
            split="test",
            transform=mnist_transforms,
        )

    def train_dataloader(self) -> DataLoader:
        # Put on GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        values = torch.cat([e[0][None, :] for e in self.train_dataset], dim=0)
        targets = torch.tensor([e[1] for e in self.train_dataset])
        self.train_dataset = TensorDataset(values.to(device), targets.to(device))
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )


if __name__ == "__main__":

    dm = SVHNDataModule(16, 0)
    dm.setup()

    # Observe sample
    x, y = next(iter(dm.train_dataloader()))

    show_samples = True
    if show_samples:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        print(x[0].permute(1, 2, 0).shape)
        ax.imshow(x[0].permute(1, 2, 0), cmap="binary")
        print(f"Displayed: `{y[0]}`")

        plt.show()

    # Info about the data
    compute_info = False
    if compute_info:
        dm = SVHNDataModule(512, 0)
        dm.setup()

        x_train = []
        for idx, batch in enumerate(dm.train_dataloader()):
            x, _ = batch
            x_train.append(x)
        for idx, batch in enumerate(dm.val_dataloader()):
            x, _ = batch
            x_train.append(x)

        x_train = torch.cat(x_train, dim=0)
        print(f"Dataset shape: {x_train.shape}")
        print(f"Dataset range: {(x_train.min(), x_train.max())}")
        print(
            f"Dataset per channel mean and std: {(x_train.mean(dim=(0,2,3)), x_train.std(dim=(0,2,3)))}"
        )
