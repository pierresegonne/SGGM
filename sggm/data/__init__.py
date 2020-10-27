from sggm.data.toy import ToyDataModule
from sggm.data.toy_2d import Toy2DDataModule
from sggm.data.uci_superconduct import UCISuperConductDataModule
from sggm.definitions import TOY, TOY_2D, UCI_SUPERCONDUCT

datamodules = {
    TOY: ToyDataModule,
    TOY_2D: Toy2DDataModule,
    UCI_SUPERCONDUCT: UCISuperConductDataModule,
}