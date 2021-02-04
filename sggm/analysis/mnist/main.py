import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as tcd

from pytorch_lightning import seed_everything, Trainer
from torch import no_grad

from sggm.data.mnist import MNISTDataModule, MNISTDataModule2D
from sggm.data.fashion_mnist import FashionMNISTDataModule
from sggm.data.not_mnist import NotMNISTDataModule
from sggm.definitions import (
    MNIST,
    MNIST_2D,
    FASHION_MNIST,
    NOT_MNIST,
)
from sggm.definitions import (
    DIGITS,
)
from sggm.vae_model import V3AE, VanillaVAE
from sggm.vae_model_helper import batch_flatten, batch_reshape
from sggm.styles_ import colours, colours_rgb
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
        print(x_var[n, :].min(), x_var[n, :].max())
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
    if isinstance(model, VanillaVAE):
        μ_z, std_z = model.decoder_μ(z), model.decoder_std(z)
        _, p_x_z = model.sample_generative(μ_z, std_z)
    elif isinstance(model, V3AE):
        batch_size = z.shape[1]
        z = torch.reshape(z, [-1, *model.latent_dims])
        μ_z, α_z, β_z = model.decoder_μ(z), model.decoder_α(z), model.decoder_β(z)
        μ_z = torch.reshape(μ_z, [-1, batch_size, model.input_size])
        α_z = torch.reshape(α_z, [-1, batch_size, model.input_size])
        β_z = torch.reshape(α_z, [-1, batch_size, model.input_size])
        _, p_x_z = model.sample_generative(μ_z, α_z, β_z)
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


def show_2d_latent_space(model, x, y):
    digits = torch.unique(y)
    colour_digits = [
        colours_rgb["primaryRed"],
        colours_rgb["green"],
    ]
    with torch.no_grad():
        if isinstance(model, VanillaVAE):
            _, _, z, _, _ = model._run_step(x)
        elif isinstance(model, V3AE):
            _, _, _, _, _, z, q_z_x, _ = model._run_step(x)
            # keep only a single z sample
            z = z[0]

    # Show specific regions
    ctr = torch.Tensor([[-0.012, -0.063]])
    lft = torch.Tensor([[-1.258, 0.416]])
    rgt = torch.Tensor([[1.17, -0.07]])
    idx = torch.norm(z - rgt, dim=1) < 0.2
    z_display = z[idx]
    fig, [ax1, ax2, ax3, ax4] = plt.subplots(1, 4)
    x_og = x[idx][0]
    with torch.no_grad():
        _, p_x = model(x_og)
    x_hat = batch_reshape(p_x.sample(), model.input_dims)
    x_mu = batch_reshape(p_x.mean, model.input_dims)
    x_var = batch_reshape(p_x.variance, model.input_dims)

    ax1.imshow(x_og[0], cmap="binary", vmin=0, vmax=1)
    ax2.imshow(x_mu[0][0], cmap="binary", vmin=0, vmax=1)
    ax3.imshow(x_var[0][0], cmap="binary")
    ax4.imshow(x_hat[0][0], cmap="binary", vmin=0, vmax=1)
    plt.show()

    fig, ax = plt.subplots()
    # Show imshow for variance -> inspiration from aleatoric_epistemic_split
    extent = 3.5
    x_mesh = torch.linspace(-extent, extent, 300)
    y_mesh = torch.linspace(-extent, extent, 300)
    x_mesh, y_mesh = torch.meshgrid(x_mesh, y_mesh)
    z_latent_mesh = torch.cat(
        (x_mesh.flatten()[:, None], y_mesh.flatten()[:, None]), dim=1
    )
    with torch.no_grad():
        if isinstance(model, VanillaVAE):
            var = model.decoder_std(z_latent_mesh)
        if isinstance(model, V3AE):
            var = model.decoder_β(z_latent_mesh) / (model.decoder_α(z_latent_mesh) - 1)
            # var_epistemic = model.decoder_α(z_latent_mesh) / (
            #     model.decoder_α(z_latent_mesh) - 1
            # )
            # var_aleatoric = model.decoder_β(z_latent_mesh) / model.decoder_α(
            #     z_latent_mesh
            # )
    # Accumulated gradient over all output cf nicki and martin
    var = torch.mean(var, dim=1)
    # reshape to x_shape
    var = var.reshape(*x_mesh.shape)
    # vmin=-1.5, vmax=2.5
    varimshw = ax.imshow(
        var,
        extent=(-extent, extent, -extent, extent),
        vmax=np.percentile(var.flatten(), 75),
        cmap="magma",
    )

    # Show two digits separately
    for i, d in enumerate(digits):
        ax.plot(
            z[:, 0][y == d],
            z[:, 1][y == d],
            "o",
            markersize=3.5,
            markerfacecolor=(*colour_digits[i], 0.9),
            markeredgewidth=1.2,
            markeredgecolor=(*colours_rgb["white"], 0.5),
            label=f"Digit {d}",
        )

    # Pseudo-inputs
    legend_ncols = 2
    show_pi = False
    if (
        isinstance(model, V3AE)
        and getattr(model, "ood_z_generation_method", None) is not None
        and show_pi
    ):
        # mult = getattr(model, "kde_bandwidth_multiplier", 10)
        # [n_mc_samples, BS, *self.latent_dims]
        z_out = model.generate_z_out(q_z_x, averaged_std=False)
        ax.plot(
            z_out[0, :, 0],
            z_out[0, :, 1],
            "o",
            markersize=3.5,
            markerfacecolor=(*colours_rgb["purple"], 0.9),
            markeredgewidth=1.2,
            markeredgecolor=(*colours_rgb["white"], 0.5),
            label="PI",
        )
        legend_ncols = 3

    # Misc
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([-extent, extent])
    ax.set_ylim([-extent, extent])
    fig.colorbar(varimshw, ax=ax)
    ax.legend(
        fancybox=True,
        shadow=False,
        ncol=legend_ncols,
        bbox_to_anchor=(0.89, 1.15),
    )

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
        bs = 512
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
    bs = 1024
    experiment_name = experiment_log.experiment_name
    misc = experiment_log.best_version.misc
    # if "seed" in misc:
    #     seed_everything(misc["seed"])
    if experiment_name == MNIST:
        dm = MNISTDataModule(bs, 0)
    if experiment_name == MNIST_2D:
        if DIGITS in misc:
            dm = MNISTDataModule2D(bs, 0, digits=misc[DIGITS])
        else:
            dm = MNISTDataModule2D(bs, 0)
    elif experiment_name == FASHION_MNIST:
        dm = FashionMNISTDataModule(bs, 0)
    elif experiment_name == NOT_MNIST:
        dm = NotMNISTDataModule(bs, 0)
    dm.setup()

    # Dataset
    training_dataset = next(iter(dm.train_dataloader()))
    x_train, y_train = training_dataset
    test_dataset = next(iter(dm.val_dataloader()))
    x_test, y_test = test_dataset

    # Reconstruction
    best_model.eval()
    with no_grad():
        x_hat_train, p_x_train = best_model(x_train)
        x_hat_test, p_x_test = best_model(x_test)

        mean_error = torch.nn.functional.mse_loss(
            batch_reshape(p_x_test.mean[0], best_model.input_dims), x_test
        )
        print(mean_error)
    #     x = batch_flatten(x_test)
    #     best_model.encoder_μ.eval()
    #     μ_x = best_model.encoder_μ(x)
    #     std_x = best_model.encoder_std(x)
    #     z, _, _ = best_model.sample_latent(μ_x, std_x)
    #     _, μ_z, α_z, β_z = best_model.parametrise_z(z)
    #     print(p_x_test.mean.shape)
    #     print(z[0][0])
    # exit()

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

    if experiment_name == MNIST_2D:
        interpolation_digits = [digits[dm.digits[0]][0], digits[dm.digits[1]][0]]
    else:
        interpolation_digits = [digits[1][0], digits[3][0]]
    fig_interpolation = plot_interpolation(best_model, *interpolation_digits)

    plt.savefig(f"{save_folder}/_interpolation.png", dpi=300)
    plt.savefig(f"{save_folder}/_interpolation.svg")
    plt.show()

    # NOTE: could be updated to if latent space is 2D
    if experiment_name == MNIST_2D:
        fig_latent_space = show_2d_latent_space(best_model, x_test, y_test)
        plt.savefig(f"{save_folder}/_latent_space.png", dpi=300)
        plt.savefig(f"{save_folder}/_latent_space.svg")
        plt.show()

    compute_all_mnist_metrics(best_model, kwargs["others"])
    plot_others_mnist(best_model, kwargs["others"], save_folder)
