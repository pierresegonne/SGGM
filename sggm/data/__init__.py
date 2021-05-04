from sggm.data.sanity_check import SanityCheckDataModule
from sggm.data.toy import ToyDataModule, ToyDataModuleShifted
from sggm.data.toy_2d import Toy2DDataModule, Toy2DDataModuleShifted
from sggm.data.uci_boston import UCIBostonDataModule, UCIBostonDataModuleShifted
from sggm.data.uci_carbon import UCICarbonDataModule, UCICarbonDataModuleShifted
from sggm.data.uci_ccpp import UCICCPPDataModule, UCICCPPDataModuleShifted
from sggm.data.uci_concrete import UCIConcreteDataModule, UCIConcreteDataModuleShifted
from sggm.data.uci_energy import UCIEnergyDataModule, UCIEnergyDataModuleShifted
from sggm.data.uci_kin8nm import UCIKin8nmDataModule, UCIKin8nmDataModuleShifted
from sggm.data.uci_naval import UCINavalDataModule, UCINavalDataModuleShifted
from sggm.data.uci_protein import UCIProteinDataModule, UCIProteinDataModuleShifted
from sggm.data.uci_superconduct import (
    UCISuperConductDataModule,
    UCISuperConductDataModuleShifted,
)
from sggm.data.uci_wine_red import UCIWineRedDataModule, UCIWineRedDataModuleShifted
from sggm.data.uci_wine_white import (
    UCIWineWhiteDataModule,
    UCIWineWhiteDataModuleShifted,
)
from sggm.data.uci_yacht import UCIYachtDataModule, UCIYachtDataModuleShifted

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
    UCI_BOSTON,
    UCI_BOSTON_SHIFTED,
    UCI_CARBON,
    UCI_CARBON_SHIFTED,
    UCI_CCPP,
    UCI_CCPP_SHIFTED,
    UCI_CONCRETE,
    UCI_CONCRETE_SHIFTED,
    UCI_ENERGY,
    UCI_ENERGY_SHIFTED,
    UCI_KIN8NM,
    UCI_KIN8NM_SHIFTED,
    UCI_NAVAL,
    UCI_NAVAL_SHIFTED,
    UCI_PROTEIN,
    UCI_PROTEIN_SHIFTED,
    UCI_SUPERCONDUCT,
    UCI_SUPERCONDUCT_SHIFTED,
    UCI_WINE_RED,
    UCI_WINE_RED_SHIFTED,
    UCI_WINE_WHITE,
    UCI_WINE_WHITE_SHIFTED,
    UCI_YACHT,
    UCI_YACHT_SHIFTED,
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
    TOY_2D: Toy2DDataModule,
    TOY_2D_SHIFTED: Toy2DDataModuleShifted,
    UCI_BOSTON: UCIBostonDataModule,
    UCI_BOSTON_SHIFTED: UCIBostonDataModuleShifted,
    UCI_CARBON: UCICarbonDataModule,
    UCI_CARBON_SHIFTED: UCICarbonDataModuleShifted,
    UCI_CCPP: UCICCPPDataModule,
    UCI_CCPP_SHIFTED: UCICCPPDataModuleShifted,
    UCI_CONCRETE: UCIConcreteDataModule,
    UCI_CONCRETE_SHIFTED: UCIConcreteDataModuleShifted,
    UCI_ENERGY: UCIEnergyDataModule,
    UCI_ENERGY_SHIFTED: UCIEnergyDataModuleShifted,
    UCI_KIN8NM: UCIKin8nmDataModule,
    UCI_KIN8NM_SHIFTED: UCIKin8nmDataModuleShifted,
    UCI_NAVAL: UCINavalDataModule,
    UCI_NAVAL_SHIFTED: UCINavalDataModuleShifted,
    UCI_PROTEIN: UCIProteinDataModule,
    UCI_PROTEIN_SHIFTED: UCIProteinDataModuleShifted,
    UCI_SUPERCONDUCT: UCISuperConductDataModule,
    UCI_SUPERCONDUCT_SHIFTED: UCISuperConductDataModuleShifted,
    UCI_WINE_RED: UCIWineRedDataModule,
    UCI_WINE_RED_SHIFTED: UCIWineRedDataModuleShifted,
    UCI_WINE_WHITE: UCIWineWhiteDataModule,
    UCI_WINE_WHITE_SHIFTED: UCIWineWhiteDataModuleShifted,
    UCI_YACHT: UCIYachtDataModule,
    UCI_YACHT_SHIFTED: UCIYachtDataModuleShifted,
    #
    CIFAR: CIFARDataModule,
    MNIST: MNISTDataModule,
    MNIST_ND: MNISTDataModuleND,
    FASHION_MNIST: FashionMNISTDataModule,
    FASHION_MNIST_ND: FashionMNISTDataModuleND,
    NOT_MNIST: NotMNISTDataModule,
    SVHN: SVHNDataModule,
}