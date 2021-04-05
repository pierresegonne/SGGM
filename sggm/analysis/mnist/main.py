import matplotlib.pyplot as plt
import torch

from pytorch_lightning import seed_everything, LightningDataModule
from torch import no_grad

from sggm.analysis.mnist.helper import (
    plot_comparison,
)
from sggm.analysis.mnist.latent_2d import (
    show_2d_latent_space,
    show_reconstruction_arbitrary_latent,
    show_reconstruction_grid,
)
from sggm.analysis.mnist.others_mnist import analyse_others_mnist
from sggm.data.mnist import MNISTDataModule, MNISTDataModuleND
from sggm.data.fashion_mnist import FashionMNISTDataModule, FashionMNISTDataModuleND
from sggm.data.not_mnist import NotMNISTDataModule
from sggm.definitions import (
    MNIST,
    MNIST_ND,
    FASHION_MNIST,
    FASHION_MNIST_ND,
    NOT_MNIST,
)
from sggm.definitions import (
    DIGITS,
)
from sggm.types_ import Union


def save_and_show(fig, name: str, show_plot: bool = True):
    fig.savefig(f"{name}.png", dpi=300)
    fig.savefig(f"{name}.svg")
    if show_plot:
        plt.show()


def get_dm(experiment_name: str, misc: dict, bs: int):
    if experiment_name == MNIST:
        dm = MNISTDataModule(bs, 0)
    elif experiment_name == MNIST_ND:
        if DIGITS in misc:
            dm = MNISTDataModuleND(bs, 0, digits=misc[DIGITS])
        else:
            dm = MNISTDataModuleND(bs, 0)
    elif experiment_name == FASHION_MNIST:
        dm = FashionMNISTDataModule(bs, 0)
    elif experiment_name == FASHION_MNIST_ND:
        if DIGITS in misc:
            dm = FashionMNISTDataModuleND(bs, 0, digits=misc[DIGITS])
        else:
            dm = FashionMNISTDataModuleND(bs, 0)
    elif experiment_name == NOT_MNIST:
        dm = NotMNISTDataModule(bs, 0)
    dm.setup()
    return dm


def plot(
    experiment_log,
    seed=False,
    show_plot=True,
    dm: Union[LightningDataModule, None] = None,
    save_folder: Union[str, None] = None,
    **kwargs,
):
    best_model = experiment_log.best_version.model
    if save_folder is None:
        save_folder = f"{experiment_log.save_dir}/{experiment_log.experiment_name}/{experiment_log.name}"

    # Get correct datamodule
    bs = 500
    experiment_name = experiment_log.experiment_name
    misc = experiment_log.best_version.misc
    if ("seed" in misc) & seed:
        seed_everything(misc["seed"])
    if dm is None:
        dm = get_dm(experiment_name, misc, bs)

    # Dataset
    training_dataset = next(iter(dm.train_dataloader()))
    x_train, y_train = training_dataset
    test_dataset = next(iter(dm.val_dataloader()))
    x_test, y_test = test_dataset

    # =====================
    # Plot

    # Reconstruction
    best_model.eval()
    # %
    best_model.dm = dm
    with no_grad():
        x_hat_train, p_x_train = best_model(x_train)
        x_hat_test, p_x_test = best_model(x_test)

    # # Reconstruction plots
    n_display = 5
    save_and_show(
        plot_comparison(n_display, x_test, p_x_test, best_model.input_dims),
        f"{save_folder}/_main",
        show_plot=show_plot,
    )

    # # Interpolation
    # interpolation_digits = get_interpolation_digits(
    #     dm, experiment_name, target_digits=[2, 5], target_digits_idx=[3, 3]
    # )
    # save_and_show(
    #     plot_interpolation(best_model, *interpolation_digits),
    #     f"{save_folder}/_interpolation",
    # )

    # 2D Latent space
    if best_model.latent_size == 2:
        # Arbitrary latent code
        z_star = torch.Tensor([[[-3.5, 3.5]]])
        (
            grid_samples,
            grid_mean,
        ) = show_reconstruction_grid(best_model)
        save_and_show(
            grid_samples,
            f"{save_folder}/_grid_samples",
            show_plot=show_plot,
        )
        save_and_show(
            grid_mean,
            f"{save_folder}/_grid_mean",
            show_plot=show_plot,
        )
        save_and_show(
            show_2d_latent_space(best_model, x_test, y_test, z_star=z_star),
            f"{save_folder}/_latent_space",
            show_plot=show_plot,
        )
        save_and_show(
            show_reconstruction_arbitrary_latent(best_model, z_star),
            f"{save_folder}/_arbitrary_z_reconstruction",
            show_plot=show_plot,
        )

    for other_mnist in kwargs.get("others", []):
        save_and_show(
            analyse_others_mnist(best_model, other_mnist, n_display),
            f"{save_folder}/{other_mnist}",
            show_plot=show_plot,
        )
