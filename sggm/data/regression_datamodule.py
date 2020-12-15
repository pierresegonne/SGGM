import multiprocessing
import numpy as np
import pytorch_lightning as pl

from torch.utils.data import random_split

"""
Abstraction level for all regression datamodules
"""

N_cpus = multiprocessing.cpu_count()

import inspect


class RegressionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        n_workers: int,
        train_val_split: float = 0.9,
        test_split: float = 0.1,
    ):
        print(inspect.signature(pl.LightningDataModule.__init__))
        pl.LightningDataModule.__init__(self)
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.test_split = test_split

        self.n_workers = n_workers if n_workers is not None else N_cpus
        self.pin_memory = True if self.n_workers > 0 else False

        # To Override
        self.max_batch_iterations = None

    def check_setup(self):
        # Check that the datasets have been set up
        assert hasattr(
            self, "train_dataset"
        ), "max_batch_iterations can only be accessed once the datamodule has been setup."

    def setup_train_val_datasets(self, train_dataset):
        N_train = train_dataset.tensors[0].shape[0]
        train_size = int(N_train * self.train_val_split)
        val_size = N_train - train_size
        self.train_dataset, self.val_dataset = random_split(
            train_dataset, [train_size, val_size]
        )

    @property
    def max_epochs(self):
        self.check_setup()
        return int(
            np.ceil(
                self.max_batch_iterations
                / np.ceil(len(self.train_dataset) / self.batch_size)
            )
        )
