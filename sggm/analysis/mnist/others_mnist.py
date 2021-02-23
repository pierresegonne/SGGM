from pytorch_lightning import Trainer
from torch import no_grad

from sggm.analysis.mnist.helper import plot_comparison
from sggm.data.mnist import MNISTDataModule
from sggm.data.fashion_mnist import FashionMNISTDataModule
from sggm.data.not_mnist import NotMNISTDataModule
from sggm.definitions import (
    MNIST,
    FASHION_MNIST,
    NOT_MNIST,
)


def compute_all_mnist_metrics(model, dm):
    trainer = Trainer()
    model.eval()
    res = trainer.test(model, datamodule=dm)


def plot_other_mnist(model, dm, n_display):
    test_dataset = next(iter(dm.test_dataloader()))
    x_test, y_test = test_dataset

    model.eval()
    with no_grad():
        x_hat_test, p_x_test = model(x_test)

    return plot_comparison(n_display, x_test, p_x_test, model.input_dims)


def analyse_others_mnist(model, other_mnist, n_display):
    dm = None
    bs = 512
    if other_mnist == MNIST:
        dm = MNISTDataModule(bs, 0)
    elif other_mnist == FASHION_MNIST:
        dm = FashionMNISTDataModule(bs, 0)
    elif other_mnist == NOT_MNIST:
        dm = NotMNISTDataModule(bs, 0)
    else:
        raise NotImplementedError(f"{other_mnist} is not a correct mnist dataset")
    dm.setup()

    compute_all_mnist_metrics(model, dm)
    return plot_other_mnist(model, dm, n_display)
