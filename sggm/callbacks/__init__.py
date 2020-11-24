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
from sggm.callbacks.data_saver import DataSaver
from sggm.callbacks.loss_printer import LossPrinter


callbacks = {
    TOY: [DataSaver, LossPrinter],
    TOY_2D: [DataSaver, LossPrinter],
    UCI_CCPP: [DataSaver, LossPrinter],
    UCI_CONCRETE: [DataSaver, LossPrinter],
    UCI_SUPERCONDUCT: [DataSaver, LossPrinter],
    UCI_WINE_RED: [DataSaver, LossPrinter],
    UCI_WINE_WHITE: [DataSaver, LossPrinter],
    UCI_YACHT: [DataSaver, LossPrinter],
}
