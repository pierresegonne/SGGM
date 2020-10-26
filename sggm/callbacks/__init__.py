from sggm.definitions import TOY, TOY_2D
from sggm.callbacks.data_saver import DataSaver
from sggm.callbacks.loss_printer import LossPrinter


callbacks = {
    TOY: [DataSaver, LossPrinter],
    TOY_2D: [DataSaver, LossPrinter],
}
