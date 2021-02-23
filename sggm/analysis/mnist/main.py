import matplotlib.pyplot as plt
from pytorch_lightning import seed_everything
from torch import no_grad

from sggm.analysis.mnist.helper import (
    get_interpolation_digits,
    plot_comparison,
    plot_interpolation,
)
from sggm.analysis.mnist.latent_2d import show_2d_latent_space
from sggm.analysis.mnist.others_mnist import analyse_others_mnist
from sggm.data.mnist import MNISTDataModule, MNISTDataModule2D
from sggm.data.fashion_mnist import FashionMNISTDataModule, FashionMNISTDataModule2D
from sggm.data.not_mnist import NotMNISTDataModule
from sggm.definitions import (
    MNIST,
    MNIST_2D,
    FASHION_MNIST,
    FASHION_MNIST_2D,
    NOT_MNIST,
)
from sggm.definitions import (
    DIGITS,
)


def save_and_show(fig, name):
    fig.savefig(f"{name}.png", dpi=300)
    fig.savefig(f"{name}.svg")
    fig.show()


def plot(experiment_log, seed=False, **kwargs):
    best_model = experiment_log.best_version.model
    save_folder = f"{experiment_log.save_dir}/{experiment_log.experiment_name}/{experiment_log.name}"

    # Get correct datamodule
    bs = 1024
    experiment_name = experiment_log.experiment_name
    misc = experiment_log.best_version.misc
    if ("seed" in misc) & seed:
        seed_everything(misc["seed"])
    if experiment_name == MNIST:
        dm = MNISTDataModule(bs, 0)
    elif experiment_name == MNIST_2D:
        if DIGITS in misc:
            dm = MNISTDataModule2D(bs, 0, digits=misc[DIGITS])
        else:
            dm = MNISTDataModule2D(bs, 0)
    elif experiment_name == FASHION_MNIST:
        dm = FashionMNISTDataModule(bs, 0)
    elif experiment_name == FASHION_MNIST_2D:
        if DIGITS in misc:
            dm = FashionMNISTDataModule2D(bs, 0, digits=misc[DIGITS])
        else:
            dm = FashionMNISTDataModule2D(bs, 0)
    elif experiment_name == NOT_MNIST:
        dm = NotMNISTDataModule(bs, 0)
    dm.setup()

    # Dataset
    training_dataset = next(iter(dm.train_dataloader()))
    x_train, y_train = training_dataset
    test_dataset = next(iter(dm.val_dataloader()))
    x_test, y_test = test_dataset

    # =====================
    # Plot

    # Reconstruction
    best_model.eval()
    with no_grad():
        x_hat_train, p_x_train = best_model(x_train)
        x_hat_test, p_x_test = best_model(x_test)

    # Reconstruction plots
    n_display = 5
    save_and_show(
        plot_comparison(n_display, x_test, p_x_test, best_model.input_dims),
        f"{save_folder}/_main",
    )

    # Interpolation
    interpolation_digits = get_interpolation_digits(
        dm, experiment_name, target_digits=[0, 5], target_digits_idx=[3, 3]
    )
    save_and_show(
        plot_interpolation(best_model, *interpolation_digits),
        f"{save_folder}/_interpolation",
    )
    exit()

    # 2D Latent space
    if experiment_name in [MNIST_2D, FASHION_MNIST_2D]:
        save_and_show(
            show_2d_latent_space(best_model, x_test, y_test),
            f"{save_folder}/_latent_space",
        )

    for other_mnist in kwargs["others"]:
        save_and_show(
            analyse_others_mnist(best_model, other_mnist, n_display),
            f"{save_folder}/{other_mnist}",
        )
