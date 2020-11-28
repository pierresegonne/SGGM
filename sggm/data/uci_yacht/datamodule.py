import pandas as pd
import pathlib

from sggm.data.uci import UCIDataModule

DATA_FILENAME = "yacht_hydrodynamics.data"
"""
Link to get the yacht_hydrodynamics.data file: https://archive.ics.uci.edu/ml/datasets/Yacht+Hydrodynamics
"""
COLUMNS = [
    "Longitudinal position of the center of the buoyancy",
    "Prismatic coefficient",
    "Length-displacement ratio",
    "Beam-draught ratio",
    "Length-beam ratio",
    "Froude number",
    "Residuary resistance per unit weight of displacement",
]
Y_LABEL = "Residuary resistance per unit weight of displacement"


class UCIYachtDataModule(UCIDataModule):
    def __init__(
        self,
        batch_size: int,
        n_workers: int,
        train_val_split: float = 0.9,
        test_split: float = 0.1,
        **kwargs,
    ):
        super(UCIYachtDataModule, self).__init__(
            batch_size,
            n_workers,
            train_val_split,
            test_split,
            **kwargs,
        )

        # Manual as we know it
        self.dims = 6
        self.out_dims = 1

    def setup(self, stage: str = None):

        df = pd.read_fwf(
            f"{pathlib.Path(__file__).parent.absolute()}/{DATA_FILENAME}",
            names=COLUMNS,
        )
        # Split features, targets
        x = df.drop(columns=[Y_LABEL]).values
        y = df[Y_LABEL].values

        super(UCIYachtDataModule, self).setup(x, y)
