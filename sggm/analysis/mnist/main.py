import matplotlib
import matplotlib.pyplot as plt

from torch import no_grad

from sggm.data.mnist import MNISTDataModule
from sggm.data.fashion_mnist import FashionMNISTDataModule
from sggm.data.not_mnist import NotMNISTDataModule

from sggm.definitions import (
    MNIST,
    FASHION_MNIST,
    NOT_MNIST,
)


def disable_ticks(ax: matplotlib.axes.Axes) -> matplotlib.axes.Axes:
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax.tick_params(axis="y", which="both", left=False, labelleft=False)
    return ax


def plot(experiment_log):
    best_model = experiment_log.best_version.model

    # Get correct datamodule
    bs = 16
    experiment_name = experiment_log.experiment_name
    if experiment_name == MNIST:
        dm = MNISTDataModule(bs, 0)
    elif experiment_name == FASHION_MNIST:
        dm = FashionMNISTDataModule(bs, 0)
    elif experiment_name == NOT_MNIST:
        dm = NotMNISTDataModule(bs, 0)
    dm.setup()

    # Dataset
    training_dataset = next(iter(dm.train_dataloader()))
    x_train, y_train = training_dataset
    test_dataset = next(iter(dm.test_dataloader()))
    x_test, y_test = test_dataset

    # Reconstruction
    with no_grad():
        x_hat_train = best_model(x_train)
        x_hat_test = best_model(x_test)

    # Figures
    n_display = 3
    train_fig = plt.figure(figsize=(8, 6))
    train_fig.suptitle("Train")
    train_gs = train_fig.add_gridspec(n_display, 2)
    train_gs.update(wspace=0.05, hspace=0.05)

    for n in range(n_display):
        for k in range(2):
            train_ax = plt.subplot(train_gs[n, k])
            # Original
            if k == 0:
                train_ax.imshow(x_train[n, :][0], cmap="binary")
            # Reconstructed
            elif k == 1:
                train_ax.imshow(x_hat_train[n, :][0], cmap="binary")
            train_ax = disable_ticks(train_ax)

            if n == n_display - 1:
                if k == 0:
                    train_ax.set_xlabel("Original")
                elif k == 1:
                    train_ax.set_xlabel("Reconstructed")

    test_fig = plt.figure(figsize=(8, 6))
    test_fig.suptitle("Test")
    test_gs = test_fig.add_gridspec(n_display, 2)
    test_gs.update(wspace=0.05, hspace=0.05)

    for n in range(n_display):
        for k in range(2):
            test_ax = plt.subplot(test_gs[n, k])
            # Original
            if k == 0:
                test_ax.imshow(x_test[n, :][0], cmap="binary")
            # Reconstructed
            elif k == 1:
                test_ax.imshow(x_hat_test[n, :][0], cmap="binary")
            test_ax = disable_ticks(test_ax)

            if n == n_display - 1:
                if k == 0:
                    test_ax.set_xlabel("Original")
                elif k == 1:
                    test_ax.set_xlabel("Reconstructed")

    plt.tight_layout()

    plt.show()
