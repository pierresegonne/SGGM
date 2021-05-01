from os import sep
import pandas as pd
import pathlib

from sggm.data.uci import UCIDataModule
from sggm.data.shifted import DataModuleShifted

DATA_FILENAME = "data.txt"
"""
Link to get the data.txt file: https://archive.ics.uci.edu/ml/machine-learning-databases/00316/
"""


class UCINavalDataModule(UCIDataModule):
    def __init__(
        self,
        batch_size: int,
        n_workers: int,
        train_val_split: float = 0.9,
        test_split: float = 0.1,
        **kwargs,
    ):
        UCIDataModule.__init__(
            self,
            batch_size,
            n_workers,
            train_val_split,
            test_split,
        )

        # Manual as we know it
        self.dims = 16
        self.out_dims = 2

    def setup(self, stage: str = None):

        df = pd.read_csv(
            f"{pathlib.Path(__file__).parent.absolute()}/{DATA_FILENAME}",
            header=None,
            sep="   ",
        )
        x = df.values[:, :-2]
        y = df.values[:, -2:]

        UCIDataModule.setup(self, x, y)


class UCINavalDataModuleShifted(UCINavalDataModule, DataModuleShifted):
    def __init__(
        self,
        batch_size: int,
        n_workers: int,
        train_val_split: float = 0.9,
        test_split: float = 0.1,
        shifting_proportion_total: float = 0.1,
        shifting_proportion_k: float = 1e-2,
        **kwargs,
    ):
        UCINavalDataModule.__init__(
            self,
            batch_size,
            n_workers,
            train_val_split,
            test_split,
        )
        DataModuleShifted.__init__(
            self, shifting_proportion_total, shifting_proportion_k
        )

    def setup(self, stage: str = None):
        UCINavalDataModule.setup(self, stage)
        DataModuleShifted.setup(self)


if __name__ == "__main__":

    dm = UCINavalDataModule(1024, 0)
    dm.setup()

    print(dm.max_epochs, dm.max_batch_iterations)
