from sggm.data.sanity_check import SanityCheckDataModule
from sggm.data.toy import ToyDataModule, ToyDataModuleShifted
from sggm.data.toy_symmetrical import ToySymmetricalDataModule
from sggm.data.toy_2d import Toy2DDataModule, Toy2DDataModuleShifted
from sggm.data.uci_boston import (
    UCIBostonDataModule,
    UCIBostonDataModuleShifted,
    UCIBostonDataModuleShiftedSplit,
)
from sggm.data.uci_carbon import (
    UCICarbonDataModule,
    UCICarbonDataModuleShifted,
    UCICarbonDataModuleShiftedSplit,
)
from sggm.data.uci_ccpp import (
    UCICCPPDataModule,
    UCICCPPDataModuleShifted,
    UCICCPPDataModuleShiftedSplit,
)
from sggm.data.uci_concrete import (
    UCIConcreteDataModule,
    UCIConcreteDataModuleShifted,
    UCIConcreteDataModuleShiftedSplit,
)
from sggm.data.uci_energy import (
    UCIEnergyDataModule,
    UCIEnergyDataModuleShifted,
    UCIEnergyDataModuleShiftedSplit,
)
from sggm.data.uci_kin8nm import (
    UCIKin8nmDataModule,
    UCIKin8nmDataModuleShifted,
    UCIKin8nmDataModuleShiftedSplit,
)
from sggm.data.uci_naval import (
    UCINavalDataModule,
    UCINavalDataModuleShifted,
    UCINavalDataModuleShiftedSplit,
)
from sggm.data.uci_protein import (
    UCIProteinDataModule,
    UCIProteinDataModuleShifted,
    UCIProteinDataModuleShiftedSplit,
)
from sggm.data.uci_superconduct import (
    UCISuperConductDataModule,
    UCISuperConductDataModuleShifted,
    UCISuperConductDataModuleShiftedSplit,
)
from sggm.data.uci_wine_red import (
    UCIWineRedDataModule,
    UCIWineRedDataModuleShifted,
    UCIWineRedDataModuleShiftedSplit,
)
from sggm.data.uci_wine_white import (
    UCIWineWhiteDataModule,
    UCIWineWhiteDataModuleShifted,
    UCIWineWhiteDataModuleShiftedSplit,
)
from sggm.data.uci_yacht import (
    UCIYachtDataModule,
    UCIYachtDataModuleShifted,
    UCIYachtDataModuleShiftedSplit,
)

from sggm.data.cifar import CIFARDataModule
from sggm.data.mnist import MNISTDataModule, MNISTDataModuleND
from sggm.data.fashion_mnist import FashionMNISTDataModule, FashionMNISTDataModuleND
from sggm.data.not_mnist import NotMNISTDataModule
from sggm.data.svhn import SVHNDataModule

from sggm.definitions import (
    SANITY_CHECK,
    TOY,
    TOY_SHIFTED,
    TOY_2D,
    TOY_2D_SHIFTED,
    TOY_SYMMETRICAL,
    UCI_BOSTON,
    UCI_BOSTON_SHIFTED,
    UCI_BOSTON_SHIFTED_SPLIT,
    UCI_CARBON,
    UCI_CARBON_SHIFTED,
    UCI_CARBON_SHIFTED_SPLIT,
    UCI_CCPP,
    UCI_CCPP_SHIFTED,
    UCI_CCPP_SHIFTED_SPLIT,
    UCI_CONCRETE,
    UCI_CONCRETE_SHIFTED,
    UCI_CONCRETE_SHIFTED_SPLIT,
    UCI_ENERGY,
    UCI_ENERGY_SHIFTED,
    UCI_ENERGY_SHIFTED_SPLIT,
    UCI_KIN8NM,
    UCI_KIN8NM_SHIFTED,
    UCI_KIN8NM_SHIFTED_SPLIT,
    UCI_NAVAL,
    UCI_NAVAL_SHIFTED,
    UCI_NAVAL_SHIFTED_SPLIT,
    UCI_PROTEIN,
    UCI_PROTEIN_SHIFTED,
    UCI_PROTEIN_SHIFTED_SPLIT,
    UCI_SUPERCONDUCT,
    UCI_SUPERCONDUCT_SHIFTED,
    UCI_SUPERCONDUCT_SHIFTED_SPLIT,
    UCI_WINE_RED,
    UCI_WINE_RED_SHIFTED,
    UCI_WINE_RED_SHIFTED_SPLIT,
    UCI_WINE_WHITE,
    UCI_WINE_WHITE_SHIFTED,
    UCI_WINE_WHITE_SHIFTED_SPLIT,
    UCI_YACHT,
    UCI_YACHT_SHIFTED,
    UCI_YACHT_SHIFTED_SPLIT,
    #
    CIFAR,
    MNIST,
    MNIST_ND,
    FASHION_MNIST,
    FASHION_MNIST_ND,
    NOT_MNIST,
    SVHN,
)

datamodules = {
    SANITY_CHECK: SanityCheckDataModule,
    TOY: ToyDataModule,
    TOY_SHIFTED: ToyDataModuleShifted,
    TOY_SYMMETRICAL: ToySymmetricalDataModule,
    TOY_2D: Toy2DDataModule,
    TOY_2D_SHIFTED: Toy2DDataModuleShifted,
    UCI_BOSTON: UCIBostonDataModule,
    UCI_BOSTON_SHIFTED: UCIBostonDataModuleShifted,
    UCI_BOSTON_SHIFTED_SPLIT: UCIBostonDataModuleShiftedSplit,
    UCI_CARBON: UCICarbonDataModule,
    UCI_CARBON_SHIFTED: UCICarbonDataModuleShifted,
    UCI_CARBON_SHIFTED_SPLIT: UCICarbonDataModuleShiftedSplit,
    UCI_CCPP: UCICCPPDataModule,
    UCI_CCPP_SHIFTED: UCICCPPDataModuleShifted,
    UCI_CCPP_SHIFTED_SPLIT: UCICCPPDataModuleShiftedSplit,
    UCI_CONCRETE: UCIConcreteDataModule,
    UCI_CONCRETE_SHIFTED: UCIConcreteDataModuleShifted,
    UCI_CONCRETE_SHIFTED_SPLIT: UCIConcreteDataModuleShiftedSplit,
    UCI_ENERGY: UCIEnergyDataModule,
    UCI_ENERGY_SHIFTED: UCIEnergyDataModuleShifted,
    UCI_ENERGY_SHIFTED_SPLIT: UCIEnergyDataModuleShiftedSplit,
    UCI_KIN8NM: UCIKin8nmDataModule,
    UCI_KIN8NM_SHIFTED: UCIKin8nmDataModuleShifted,
    UCI_KIN8NM_SHIFTED_SPLIT: UCIKin8nmDataModuleShiftedSplit,
    UCI_NAVAL: UCINavalDataModule,
    UCI_NAVAL_SHIFTED: UCINavalDataModuleShifted,
    UCI_NAVAL_SHIFTED_SPLIT: UCINavalDataModuleShiftedSplit,
    UCI_PROTEIN: UCIProteinDataModule,
    UCI_PROTEIN_SHIFTED: UCIProteinDataModuleShifted,
    UCI_PROTEIN_SHIFTED_SPLIT: UCIProteinDataModuleShiftedSplit,
    UCI_SUPERCONDUCT: UCISuperConductDataModule,
    UCI_SUPERCONDUCT_SHIFTED: UCISuperConductDataModuleShifted,
    UCI_SUPERCONDUCT_SHIFTED_SPLIT: UCISuperConductDataModuleShiftedSplit,
    UCI_WINE_RED: UCIWineRedDataModule,
    UCI_WINE_RED_SHIFTED: UCIWineRedDataModuleShifted,
    UCI_WINE_RED_SHIFTED_SPLIT: UCIWineRedDataModuleShiftedSplit,
    UCI_WINE_WHITE: UCIWineWhiteDataModule,
    UCI_WINE_WHITE_SHIFTED: UCIWineWhiteDataModuleShifted,
    UCI_WINE_WHITE_SHIFTED_SPLIT: UCIWineWhiteDataModuleShiftedSplit,
    UCI_YACHT: UCIYachtDataModule,
    UCI_YACHT_SHIFTED: UCIYachtDataModuleShifted,
    UCI_YACHT_SHIFTED_SPLIT: UCIYachtDataModuleShiftedSplit,
    #
    CIFAR: CIFARDataModule,
    MNIST: MNISTDataModule,
    MNIST_ND: MNISTDataModuleND,
    FASHION_MNIST: FashionMNISTDataModule,
    FASHION_MNIST_ND: FashionMNISTDataModuleND,
    NOT_MNIST: NotMNISTDataModule,
    SVHN: SVHNDataModule,
}