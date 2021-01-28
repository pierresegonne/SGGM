import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from pytorch_lightning import Trainer
from torch import no_grad

from sggm.data.mnist import MNISTDataModule
from sggm.data.fashion_mnist import FashionMNISTDataModule
from sggm.data.not_mnist import NotMNISTDataModule
from sggm.definitions import (
    MNIST,
    FASHION_MNIST,
    NOT_MNIST,
)
from sggm.vae_model_helper import batch_flatten, batch_reshape
from sggm.types_ import List


def disable_ticks(ax: matplotlib.axes.Axes) -> matplotlib.axes.Axes:
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax.tick_params(axis="y", which="both", left=False, labelleft=False)
    return ax


def plot_comparison(n_display, title, x_og, p_x, input_dims):
    fig = plt.figure(figsize=(6, 6))  # constrained_layout=True
    # fig.suptitle(title)
    gs = fig.add_gridspec(
        4, n_display, width_ratios=[1] * n_display, height_ratios=[1, 1, 1, 1]
    )

    gs.update(wspace=0, hspace=0)

    x_hat = batch_reshape(p_x.sample(), input_dims)
    x_mu = batch_reshape(p_x.mean, input_dims)
    x_var = batch_reshape(p_x.variance, input_dims)

    for n in range(n_display):
        for k in range(4):
            ax = plt.subplot(gs[k, n])
            ax = disable_ticks(ax)
            # Original
            if k == 0:
                ax.imshow(x_og[n, :][0], cmap="binary", vmin=0, vmax=1)
            # Mean
            elif k == 1:
                ax.imshow(x_mu[n, :][0], cmap="binary", vmin=0, vmax=1)
            # Variance
            elif k == 2:
                ax.imshow(x_var[n, :][0], cmap="binary")
            # Sample
            elif k == 3:
                ax.imshow(x_hat[n, :][0], cmap="binary", vmin=0, vmax=1)

    return fig


def input_to_latent(model, x):
    x = batch_flatten(x)
    μ_x = model.encoder_μ(x)
    std_x = model.encoder_std(x)
    z, _, _ = model.sample_latent(μ_x, std_x)
    return z


def latent_to_mean(model, z):
    μ_z, std_z = model.decoder_μ(z), model.decoder_std(z)
    _, p_x_z = model.sample_generative(μ_z, std_z)
    return batch_reshape(p_x_z.mean, model.input_dims)


def interpolation(tau, model, img1, img2):

    with no_grad():
        # latent vector of first image
        latent_1 = input_to_latent(model, img1)

        # latent vector of second image
        latent_2 = input_to_latent(model, img2)

        # interpolation of the two latent vectors
        inter_latent = tau * latent_1 + (1 - tau) * latent_2

        # reconstruct interpolated image
        inter_image = latent_to_mean(model, inter_latent).numpy()

    return inter_image


def plot_interpolation(model, img1, img2):
    tau_range = np.flip(np.linspace(0, 1, 10))

    fig = plt.figure()  # constrained_layout=True
    # fig.suptitle(title)
    gs = fig.add_gridspec(2, 5, width_ratios=[1] * 5, height_ratios=[1, 1])
    gs.update(hspace=0.5, wspace=0.001)

    for i, tau in enumerate(tau_range):
        row, col = i // 5, i % 5
        ax = plt.subplot(gs[row, col])
        ax = disable_ticks(ax)

        interpolated_img = interpolation(float(tau), model, img1, img2)

        ax.imshow(interpolated_img[0][0], cmap="binary")
        ax.set_title(r"$\tau$=" + str(round(tau, 1)))
    return fig


def compute_all_mnist_metrics(model, others_mnist: List[str]):
    if others_mnist is None:
        return
    for other in others_mnist:
        ds = None
        bs = 256
        if other == FASHION_MNIST:
            ds = FashionMNISTDataModule(bs, 0)
        elif other == NOT_MNIST:
            ds = NotMNISTDataModule(bs, 0)
        else:
            raise NotImplementedError(f"{other} is not a correct mnist dataset")
        ds.setup()

        trainer = Trainer()
        model.eval()
        res = trainer.test(model, datamodule=ds)
        print(f"  [{other}]:", res)


def plot_others_mnist(model, others_mnist: List[str], save_folder):
    if others_mnist is None:
        return
    for other in others_mnist:
        dm = None
        bs = 16
        if other == FASHION_MNIST:
            dm = FashionMNISTDataModule(bs, 0)
        elif other == NOT_MNIST:
            dm = NotMNISTDataModule(bs, 0)
        else:
            raise NotImplementedError(f"{other} is not a correct mnist dataset")
        dm.setup()

        test_dataset = next(iter(dm.test_dataloader()))
        x_test, y_test = test_dataset

        model.eval()
        with no_grad():
            x_hat_test, p_x_test = model(x_test)

        plot_comparison(5, "Test", x_test, p_x_test, model.input_dims)
        plt.savefig(f"{save_folder}/_main_{other}.png", dpi=300)
        plt.savefig(f"{save_folder}/_main_{other}.svg")
        plt.show()


def plot(experiment_log, **kwargs):
    best_model = experiment_log.best_version.model
    save_folder = f"{experiment_log.save_dir}/{experiment_log.experiment_name}/{experiment_log.name}"

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
    best_model.eval()
    with no_grad():
        x_hat_train, p_x_train = best_model(x_train)
        x_hat_test, p_x_test = best_model(x_test)

    # Figures
    n_display = 5

    fig_test = plot_comparison(
        n_display, "Test", x_test, p_x_test, best_model.input_dims
    )
    plt.savefig(f"{save_folder}/_main.png", dpi=300)
    plt.savefig(f"{save_folder}/_main.svg")
    plt.show()

    # Interpolation
    digits = [[] for _ in range(10)]
    for img_batch, label_batch in dm.test_dataloader():
        for i in range(img_batch.size(0)):
            digits[label_batch[i]].append(img_batch[i : i + 1])
        if sum(len(d) for d in digits) >= 100:
            break

    fig_interpolation = plot_interpolation(best_model, digits[1][0], digits[3][0])

    plt.savefig(f"{save_folder}/_interpolation.png", dpi=300)
    plt.savefig(f"{save_folder}/_interpolation.svg")
    plt.show()

    compute_all_mnist_metrics(best_model, kwargs["others"])
    plot_others_mnist(best_model, kwargs["others"], save_folder)
