import pandas as pd
import pathlib

from sggm.data.uci import UCIDataModule

DATA_FILENAME = "concrete.csv"
"""
Link to get the concrete.csv file: https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength
Note that I converted the xls file to csv, renamed the label column and the file itself.
"""
COLUMNS = [
    "Cement (component 1)(kg in a m^3 mixture)",
    "Blast Furnace Slag (component 2)(kg in a m^3 mixture)",
    "Fly Ash (component 3)(kg in a m^3 mixture)",
    "Water  (component 4)(kg in a m^3 mixture)",
    "Superplasticizer (component 5)(kg in a m^3 mixture)",
    "Coarse Aggregate  (component 6)(kg in a m^3 mixture)",
    "Fine Aggregate (component 7)(kg in a m^3 mixture)",
    "Age (day)",
]
Y_LABEL = "Concrete compressive strength(MPa)"


class UCIConcreteDataModule(UCIDataModule):
    def __init__(
        self,
        batch_size: int,
        n_workers: int,
        train_val_split: float = 0.8,
        test_split: float = 0.1,
        **kwargs,
    ):
        super(UCIConcreteDataModule, self).__init__(
            batch_size,
            n_workers,
            train_val_split,
            test_split,
            **kwargs,
        )

        # Manual as we know it
        self.dims = 8
        self.out_dims = 1

    def setup(self, stage: str = None):

        df = pd.read_csv(f"{pathlib.Path(__file__).parent.absolute()}/{DATA_FILENAME}")
        # Split features, targets
        x = df.drop(columns=[Y_LABEL]).values
        y = df[Y_LABEL].values

        super(UCIConcreteDataModule, self).setup(x, y)
