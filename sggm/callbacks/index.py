from sggm.definitions import TOY
from sggm.callbacks.data_saver import DataSaver
from sggm.callbacks.loss_printer import LossPrinter


callbacks = {
    TOY: [DataSaver, LossPrinter],
}
