import pandas as pd
import pathlib

from sggm.data.uci import UCIDataModule

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
