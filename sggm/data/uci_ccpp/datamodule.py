import pandas as pd
import pathlib

from typing import Union

from sggm.data.uci import UCIDataModule
from sggm.data.shifted import DataModuleShifted, DataModuleShiftedSplit

DATA_FILENAME = "ccpp.csv"
"""
Link to get the concrete.csv file: https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant
Note that I converted the xls file to csv, removed the sheets and renamed and the file itself.
"""
COLUMNS = ["AT", "V", "AP", "RH", "PE"]
Y_LABEL = "PE"


class UCICCPPDataModule(UCIDataModule):
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
        self.dims = 4
        self.out_dims = 1

    def setup(self, stage: str = None):

        df = pd.read_csv(f"{pathlib.Path(__file__).parent.absolute()}/{DATA_FILENAME}")
        # Split features, targets
        x = df.drop(columns=[Y_LABEL]).values
        y = df[Y_LABEL].values

        UCIDataModule.setup(self, x, y)


class UCICCPPDataModuleShifted(UCICCPPDataModule, DataModuleShifted):
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
        UCICCPPDataModule.__init__(
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
        UCICCPPDataModule.setup(self, stage)
        DataModuleShifted.setup(self)


class UCICCPPDataModuleShiftedSplit(UCICCPPDataModule, DataModuleShiftedSplit):
    def __init__(
        self,
        batch_size: int,
        n_workers: int,
        train_val_split: float = 0.9,
        test_split: float = 0.1,
        **kwargs,
    ):
        UCICCPPDataModule.__init__(
            self,
            batch_size,
            n_workers,
            train_val_split,
            test_split,
        )

    def setup(self, dim_idx: Union[None, int] = None, stage: str = None):
        UCICCPPDataModule.setup(self, stage)
        DataModuleShiftedSplit.setup(self, dim_idx, stage)


if __name__ == "__main__":

    dm = UCICCPPDataModule(1024, 0)
    dm.setup()

    # Info about the data
    import math

    print(dm.y_std)
