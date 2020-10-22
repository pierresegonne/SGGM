from sggm.definitions import TOY
from sggm.callbacks.loss_printer import LossPrinter

callbacks = {
    TOY: [LossPrinter],
}
