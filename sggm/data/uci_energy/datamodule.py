import pandas as pd
import pathlib

from sggm.data.uci import UCIDataModule
from sggm.data.shifted import DataModuleShifted

DATA_FILENAME = "energy.csv"
"""
Link to get the energy.csv file: https://archive.ics.uci.edu/ml/datasets/Energy+efficiency
Note that I converted the xlsx file to csv, and renamed the file.
"""
COLUMNS = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]
Y_LABEL = ["Y1", "Y2"]


class UCIEnergyDataModule(UCIDataModule):
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
        self.dims = 8
        self.out_dims = 2

    def setup(self, stage: str = None):

        df = pd.read_csv(f"{pathlib.Path(__file__).parent.absolute()}/{DATA_FILENAME}")
        # Split features, targets
        x = df[COLUMNS].values
        y = df[Y_LABEL].values

        UCIDataModule.setup(self, x, y)


class UCIEnergyDataModuleShifted(UCIEnergyDataModule, DataModuleShifted):
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
        UCIEnergyDataModule.__init__(
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
        UCIEnergyDataModule.setup(self, stage)
        DataModuleShifted.setup(self)


if __name__ == "__main__":

    dm = UCIEnergyDataModule(1024, 0)
    dm.setup()
