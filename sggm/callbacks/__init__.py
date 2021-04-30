from sggm.definitions import (
    SANITY_CHECK,
    TOY,
    TOY_SHIFTED,
    TOY_2D,
    TOY_2D_SHIFTED,
    UCI_CARBON,
    UCI_CARBON_SHIFTED,
    UCI_CCPP,
    UCI_CCPP_SHIFTED,
    UCI_CONCRETE,
    UCI_CONCRETE_SHIFTED,
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
from sggm.callbacks.data_saver import DataSaver
from sggm.callbacks.fit_saver import FitSaver
from sggm.callbacks.kl_saver import KLSaver
from sggm.callbacks.loss_printer import LossPrinter
from sggm.callbacks.img_generated_saver import IMGGeneratedSaver


callbacks = {
    SANITY_CHECK: [DataSaver, LossPrinter],
    TOY: [DataSaver, LossPrinter],
    TOY_SHIFTED: [DataSaver, LossPrinter],
    TOY_2D: [DataSaver, LossPrinter],
    TOY_2D_SHIFTED: [DataSaver, LossPrinter],
    UCI_CARBON: [LossPrinter],
    UCI_CARBON_SHIFTED: [LossPrinter],
    UCI_CCPP: [LossPrinter],
    UCI_CCPP_SHIFTED: [LossPrinter],
    UCI_CONCRETE: [LossPrinter],
    UCI_CONCRETE_SHIFTED: [LossPrinter],
    UCI_SUPERCONDUCT: [LossPrinter],
    UCI_SUPERCONDUCT_SHIFTED: [LossPrinter],
    UCI_WINE_RED: [LossPrinter],
    UCI_WINE_RED_SHIFTED: [LossPrinter],
    UCI_WINE_WHITE: [LossPrinter],
    UCI_WINE_WHITE_SHIFTED: [LossPrinter],
    UCI_YACHT: [LossPrinter],
    UCI_YACHT_SHIFTED: [LossPrinter],
    #
    CIFAR: [],
    MNIST: [IMGGeneratedSaver],
    MNIST_ND: [],
    FASHION_MNIST: [IMGGeneratedSaver],
    FASHION_MNIST_ND: [],
    NOT_MNIST: [IMGGeneratedSaver],
    SVHN: [],
}
