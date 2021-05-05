import pandas as pd
import pathlib

from typing import Union

from sggm.data.uci import UCIDataModule
from sggm.data.shifted import DataModuleShifted, DataModuleShiftedSplit

DATA_FILENAME = "CASP.csv"
"""
Link to get the CASP.csv file: https://archive.ics.uci.edu/ml/datasets/Physicochemical+Properties+of+Protein+Tertiary+Structure#
"""
COLUMNS = [f"F{i}" for i in range(1, 10)]
Y_LABEL = ["RMSD"]


class UCIProteinDataModule(UCIDataModule):
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
        self.dims = 9
        self.out_dims = 1

    def setup(self, stage: str = None):

        df = pd.read_csv(f"{pathlib.Path(__file__).parent.absolute()}/{DATA_FILENAME}")
        # Split features, targets
        x = df[COLUMNS].values
        y = df[Y_LABEL].values

        UCIDataModule.setup(self, x, y)


class UCIProteinDataModuleShifted(UCIProteinDataModule, DataModuleShifted):
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
        UCIProteinDataModule.__init__(
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
        UCIProteinDataModule.setup(self, stage)
        DataModuleShifted.setup(self)


class UCIProteinDataModuleShiftedSplit(UCIProteinDataModule, DataModuleShiftedSplit):
    def __init__(
        self,
        batch_size: int,
        n_workers: int,
        train_val_split: float = 0.9,
        test_split: float = 0.1,
        **kwargs,
    ):
        UCIProteinDataModule.__init__(
            self,
            batch_size,
            n_workers,
            train_val_split,
            test_split,
        )

    def setup(self, dim_idx: Union[None, int] = None, stage: str = None):
        UCIProteinDataModule.setup(self, stage)
        DataModuleShiftedSplit.setup(self, dim_idx, stage)


if __name__ == "__main__":

    dm = UCIProteinDataModule(1024, 0)
    dm.setup()
