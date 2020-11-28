import multiprocessing
import numpy as np
import pytorch_lightning as pl

"""
Abstraction level for all regression datamodules
"""

N_cpus = multiprocessing.cpu_count()


class RegressionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        n_workers: int,
        train_val_split: float = 0.9,
        test_split: float = 0.1,
        **kwargs,
    ):
        super(RegressionDataModule, self).__init__()
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

    @property
    def max_epochs(self):
        self.check_setup()
        return int(
            np.ceil(
                self.max_batch_iterations
                / np.ceil(len(self.train_dataset) / self.batch_size)
            )
        )
