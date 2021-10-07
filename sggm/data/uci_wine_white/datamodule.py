import pandas as pd
import pathlib

from typing import Union

from sggm.data.uci import UCIDataModule
from sggm.data.shifted import DataModuleShifted, DataModuleShiftedSplit

DATA_FILENAME = "winequality-white.csv"
"""
Link to get the concrete.csv file: https://archive.ics.uci.edu/ml/datasets/Wine+Quality
Note that I changed the delimiter from ; to ,
"""
COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "quality",
]
Y_LABEL = "quality"


class UCIWineWhiteDataModule(UCIDataModule):
    def __init__(
        self,
        batch_size: int,
        n_workers: int,
        train_val_split: float = 0.9,
        test_split: float = 0.1,
        **kwargs,
    ):
        super(UCIWineWhiteDataModule, self).__init__(
            batch_size,
            n_workers,
            train_val_split,
            test_split,
            **kwargs,
        )

        # Manual as we know it
        self.dims = 11
        self.out_dims = 1

    def setup(self, stage: str = None):

        df = pd.read_csv(f"{pathlib.Path(__file__).parent.absolute()}/{DATA_FILENAME}")
        # Split features, targets
        x = df.drop(columns=[Y_LABEL]).values
        y = df[Y_LABEL].values

        super(UCIWineWhiteDataModule, self).setup(x, y)


class UCIWineWhiteDataModuleShifted(UCIWineWhiteDataModule, DataModuleShifted):
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
        UCIWineWhiteDataModule.__init__(
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
        UCIWineWhiteDataModule.setup(self, stage)
        DataModuleShifted.setup(self)


class UCIWineWhiteDataModuleShiftedSplit(
    UCIWineWhiteDataModule, DataModuleShiftedSplit
):
    def __init__(
        self,
        batch_size: int,
        n_workers: int,
        train_val_split: float = 0.9,
        test_split: float = 0.1,
        **kwargs,
    ):
        UCIWineWhiteDataModule.__init__(
            self,
            batch_size,
            n_workers,
            train_val_split,
            test_split,
        )

    def setup(self, dim_idx: Union[None, int] = None, stage: str = None):
        UCIWineWhiteDataModule.setup(self, stage)
        DataModuleShiftedSplit.setup(self, dim_idx, stage)


if __name__ == "__main__":
    from sggm.definitions import STAGE_SETUP_SHIFTED_SPLIT

    dm = UCIWineWhiteDataModuleShiftedSplit(1024, 0)
    dm.setup(dim_idx=1, stage=STAGE_SETUP_SHIFTED_SPLIT)
    print(len(dm.test_dataset))
    print(len(dm.train_dataset))
