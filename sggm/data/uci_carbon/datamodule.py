import pandas as pd
import pathlib

from typing import Union

from sggm.data.uci import UCIDataModule
from sggm.data.shifted import DataModuleShifted, DataModuleShiftedSplit

DATA_FILENAME = "carbon_nanotubes.csv"
"""
"""
COLUMNS = [
    "Chiral indice n",
    "Chiral indice m",
    "Initial atomic coordinate u",
    "Initial atomic coordinate v",
    "Initial atomic coordinate w",
]
Y_LABEL = [
    "Calculated atomic coordinates u'",
    "Calculated atomic coordinates v'",
    "Calculated atomic coordinates w'",
]


class UCICarbonDataModule(UCIDataModule):
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
        self.dims = 5
        self.out_dims = 3

    def setup(self, stage: str = None):

        df = pd.read_csv(
            f"{pathlib.Path(__file__).parent.absolute()}/{DATA_FILENAME}", sep=";"
        )

        x = df.drop(columns=Y_LABEL).values
        y = df[Y_LABEL].values

        UCIDataModule.setup(self, x, y)


class UCICarbonDataModuleShifted(UCICarbonDataModule, DataModuleShifted):
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
        UCICarbonDataModule.__init__(
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
        UCICarbonDataModule.setup(self, stage)
        DataModuleShifted.setup(self)


class UCICarbonDataModuleShiftedSplit(UCICarbonDataModule, DataModuleShiftedSplit):
    def __init__(
        self,
        batch_size: int,
        n_workers: int,
        train_val_split: float = 0.9,
        test_split: float = 0.1,
        **kwargs,
    ):
        UCICarbonDataModule.__init__(
            self,
            batch_size,
            n_workers,
            train_val_split,
            test_split,
        )

    def setup(self, dim_idx: Union[None, int] = None, stage: str = None):
        UCICarbonDataModule.setup(self, stage)
        DataModuleShiftedSplit.setup(self, dim_idx, stage)


if __name__ == "__main__":

    dm = UCICarbonDataModule(1024, 0)
    dm.setup()

    print(dm.max_epochs, dm.max_batch_iterations)
