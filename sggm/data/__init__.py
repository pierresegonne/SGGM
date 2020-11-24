from sggm.data.toy import ToyDataModule
from sggm.data.toy_2d import Toy2DDataModule
from sggm.data.uci_ccpp import UCICCPPDataModule
from sggm.data.uci_concrete import UCIConcreteDataModule
from sggm.data.uci_superconduct import UCISuperConductDataModule
from sggm.data.uci_wine_red import UCIWineRedDataModule
from sggm.data.uci_wine_white import UCIWineWhiteDataModule
from sggm.data.uci_yacht import UCIYachtDataModule
from sggm.definitions import (
    TOY,
    TOY_2D,
    UCI_CCPP,
    UCI_CONCRETE,
    UCI_SUPERCONDUCT,
    UCI_WINE_RED,
    UCI_WINE_WHITE,
    UCI_YACHT,
)

datamodules = {
    TOY: ToyDataModule,
    TOY_2D: Toy2DDataModule,
    UCI_CCPP: UCICCPPDataModule,
    UCI_CONCRETE: UCIConcreteDataModule,
    UCI_SUPERCONDUCT: UCISuperConductDataModule,
    UCI_WINE_RED: UCIWineRedDataModule,
    UCI_WINE_WHITE: UCIWineWhiteDataModule,
    UCI_YACHT: UCIYachtDataModule,
}