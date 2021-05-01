import numpy as np
import pandas as pd
import pathlib

from sggm.data.uci import UCIDataModule
from sggm.data.shifted import DataModuleShifted

DATA_FILENAME = "housing.data"
"""
Link to get the housing dataset: https://archive.ics.uci.edu/ml/machine-learning-databases/housing/
"""


class UCIBostonDataModule(UCIDataModule):
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
        self.dims = 13
        self.out_dims = 1

    def setup(self, stage: str = None):
        df = pd.read_csv(f"{pathlib.Path(__file__).parent.absolute()}/{DATA_FILENAME}")
        # Loads rows as string
        data = np.empty((len(df.index), self.dims + self.out_dims))
        for i in range(data.shape[0]):
            data[i] = np.array(
                [float(el) for el in df.values[i][0].split(" ") if el != ""]
            )
        x = data[:, :-1]
        y = data[:, -1]

        UCIDataModule.setup(self, x, y)


class UCIBostonDataModuleShifted(UCIBostonDataModule, DataModuleShifted):
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
        UCIBostonDataModule.__init__(
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
        UCIBostonDataModule.setup(self, stage)
        DataModuleShifted.setup(self)


if __name__ == "__main__":

    dm = UCIBostonDataModule(1024, 0)
    dm.setup()