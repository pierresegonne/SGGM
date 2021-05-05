import pandas as pd
import pathlib

from typing import Union

from sggm.data.uci import UCIDataModule
from sggm.data.shifted import DataModuleShifted, DataModuleShiftedSplit

DATA_FILENAME = "raw.csv"
"""
Link to get the raw.csv data: https://archive.ics.uci.edu/ml/datasets/Superconductivty+Data
"""
Y_LABEL = "critical_temp"


class UCISuperConductDataModule(UCIDataModule):
    def __init__(
        self,
        batch_size: int,
        n_workers: int,
        train_val_split: float = 0.9,
        test_split: float = 0.1,
        **kwargs,
    ):
        super(UCISuperConductDataModule, self).__init__(
            batch_size,
            n_workers,
            train_val_split,
            test_split,
            **kwargs,
        )

        # Manual as we know it
        self.dims = 81
        self.out_dims = 1

    def setup(self, stage: str = None):

        df = pd.read_csv(f"{pathlib.Path(__file__).parent.absolute()}/{DATA_FILENAME}")
        # Split features, targets
        x = df.drop(columns=[Y_LABEL]).values
        y = df[Y_LABEL].values

        super(UCISuperConductDataModule, self).setup(x, y)


class UCISuperConductDataModuleShifted(UCISuperConductDataModule, DataModuleShifted):
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
        UCISuperConductDataModule.__init__(
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
        UCISuperConductDataModule.setup(self, stage)
        DataModuleShifted.setup(self)


class UCISuperConductDataModuleShiftedSplit(
    UCISuperConductDataModule, DataModuleShiftedSplit
):
    def __init__(
        self,
        batch_size: int,
        n_workers: int,
        train_val_split: float = 0.9,
        test_split: float = 0.1,
        **kwargs,
    ):
        UCISuperConductDataModule.__init__(
            self,
            batch_size,
            n_workers,
            train_val_split,
            test_split,
        )

    def setup(self, dim_idx: Union[None, int] = None, stage: str = None):
        UCISuperConductDataModule.setup(self, stage)
        DataModuleShiftedSplit.setup(self, dim_idx, stage)


if __name__ == "__main__":

    dm = UCISuperConductDataModule(1024, 0)
    dm.setup()

    # Info about the data
    print(dm.y_std)
    # 34.16975968
