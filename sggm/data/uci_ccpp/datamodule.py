import pandas as pd
import pathlib

from sggm.data.uci import UCIDataModule

DATA_FILENAME = "ccpp.csv"
"""
Link to get the concrete.csv file: https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant
Note that I converted the xls file to csv, removed the sheets and renamed and the file itself.
"""
COLUMNS = [
    "AT",
    "V",
    "AP",
    "RH",
]
Y_LABEL = "PE"


class UCICCPPDataModule(UCIDataModule):
    def __init__(
        self,
        batch_size: int,
        n_workers: int,
        train_val_split: float = 0.8,
        test_split: float = 0.1,
        **kwargs,
    ):
        super(UCICCPPDataModule, self).__init__(
            batch_size,
            n_workers,
            train_val_split,
            test_split,
            **kwargs,
        )

        # Manual as we know it
        self.dims = 4
        self.out_dims = 1

    def setup(self, stage: str = None):

        df = pd.read_csv(f"{pathlib.Path(__file__).parent.absolute()}/{DATA_FILENAME}")
        # Split features, targets
        x = df.drop(columns=[Y_LABEL]).values
        y = df[Y_LABEL].values
        print("ccpp")
        print(x.shape, y.shape)

        super(UCICCPPDataModule, self).setup(x, y)
