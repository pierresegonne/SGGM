from sggm.definitions import (
    SANITY_CHECK,
    TOY,
    TOY_SHIFTED,
    TOY_2D,
    TOY_2D_SHIFTED,
    TOY_SYMMETRICAL,
    UCI_ALL,
    #
    CIFAR,
    MNIST,
    MNIST_ND,
    EMNIST,
    FASHION_MNIST,
    FASHION_MNIST_ND,
    KMNIST,
    NOT_MNIST,
    STL10,
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
    TOY_SYMMETRICAL: [DataSaver, LossPrinter],
    TOY_2D: [DataSaver, LossPrinter],
    TOY_2D_SHIFTED: [DataSaver, LossPrinter],
    #
    CIFAR: [],
    MNIST: [IMGGeneratedSaver],
    MNIST_ND: [],
    EMNIST: [IMGGeneratedSaver],
    FASHION_MNIST: [IMGGeneratedSaver],
    FASHION_MNIST_ND: [],
    KMNIST: [IMGGeneratedSaver],
    NOT_MNIST: [IMGGeneratedSaver],
    STL10: [],
    SVHN: [],
}

for uci in UCI_ALL:
    callbacks[uci] = [LossPrinter]
