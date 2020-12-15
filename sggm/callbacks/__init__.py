from sggm.definitions import (
    TOY,
    TOY_SHIFTED,
    TOY_2D,
    UCI_CCPP,
    UCI_CONCRETE,
    UCI_SUPERCONDUCT,
    UCI_WINE_RED,
    UCI_WINE_WHITE,
    UCI_YACHT,
    #
    MNIST,
    FASHION_MNIST,
    NOT_MNIST,
)
from sggm.callbacks.data_saver import DataSaver
from sggm.callbacks.fit_saver import FitSaver
from sggm.callbacks.kl_saver import KLSaver
from sggm.callbacks.loss_printer import LossPrinter


callbacks = {
    TOY: [DataSaver, LossPrinter],
    TOY_SHIFTED: [DataSaver, LossPrinter],
    TOY_2D: [DataSaver, LossPrinter],
    UCI_CCPP: [LossPrinter],
    UCI_CONCRETE: [LossPrinter],
    UCI_SUPERCONDUCT: [LossPrinter],
    UCI_WINE_RED: [LossPrinter],
    UCI_WINE_WHITE: [LossPrinter],
    UCI_YACHT: [LossPrinter],
    #
    MNIST: [],
    FASHION_MNIST: [],
    NOT_MNIST: [],
}
