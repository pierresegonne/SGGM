import pandas as pd
import pathlib

from sggm.data.uci import UCIDataModule
from sggm.data.shifted import DataModuleShifted

DATA_FILENAME = "kin8nm.csv"
"""
Link to get the kin8nm.csv file: https://www.openml.org/d/189
"""
COLUMNS = [
    "theta1",
    "theta2",
    "theta3",
    "theta4",
    "theta5",
    "theta6",
    "theta7",
    "theta8",
]
Y_LABEL = "y"


class UCIKin8nmDataModule(UCIDataModule):
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
        self.out_dims = 1

    def setup(self, stage: str = None):

        df = pd.read_csv(f"{pathlib.Path(__file__).parent.absolute()}/{DATA_FILENAME}")
        # Split features, targets
        x = df[COLUMNS].values
        y = df[Y_LABEL].values

        UCIDataModule.setup(self, x, y)


class UCIKin8nmDataModuleShifted(UCIKin8nmDataModule, DataModuleShifted):
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
        UCIKin8nmDataModule.__init__(
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
        UCIKin8nmDataModule.setup(self, stage)
        DataModuleShifted.setup(self)


if __name__ == "__main__":

    dm = UCIKin8nmDataModule(1024, 0)
    dm.setup()

    print(dm.max_epochs, dm.max_batch_iterations)
