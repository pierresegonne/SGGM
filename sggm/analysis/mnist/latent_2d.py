from typing import Tuple, Union

from geoml.curve import BasicCurve, DiscreteCurve
from geoml.manifold import EmbeddedManifold
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.distributions as D
from torchvision import transforms
from torchvision.utils import make_grid

from sggm.analysis.utils import disable_ticks
from sggm.vae_model import BaseVAE, VanillaVAE
from sggm.vae_model_manifold import VanillaVAEm
from sggm.v3ae_model import V3AE
from sggm.v3ae_model_manifold import V3AEm
from sggm.vae_model_helper import batch_reshape
from sggm.styles_ import colours, colours_rgb, random_rgb_colour

colour_digits = [
    (1 / 255, 133 / 255, 90 / 255),
    (200 / 255, 154 / 255, 1 / 255),
]

colour_digits += [random_rgb_colour() for _ in range(10)]


def decode_from_latent(
    model: BaseVAE, z: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(model, VanillaVAE):
        μ_z, std_z = (
            model.decoder_μ(z.reshape(-1, model.latent_size)),
            model.decoder_std(z.reshape(-1, model.latent_size)),
        )
        x_hat, p_x = model.sample_generative(μ_z, std_z)
    elif isinstance(model, V3AE):
        _, μ_z, α_z, β_z = model.parametrise_z(z)
        x_hat, p_x = model.sample_generative(μ_z, α_z, β_z)

    x_hat = batch_reshape(x_hat, model.input_dims)
    x_mu = batch_reshape(p_x.mean, model.input_dims)
    x_var = batch_reshape(p_x.variance, model.input_dims)

    return x_hat, x_mu, x_var


def show_reconstruction_arbitrary_latent(
    model: pl.LightningModule, z_star: torch.Tensor
) -> plt.Figure:
    x_hat, x_mu, x_var = decode_from_latent(model, z_star)

    fig, (ax_reconstruct, ax_mean, ax_std) = plt.subplots(3, 1)
    # Reconstruction
    ax_reconstruct.imshow(x_hat[0, :][0], cmap="binary", vmin=0, vmax=1)
    # Mean
    ax_mean.imshow(x_mu[0, :][0], cmap="binary", vmin=0, vmax=1)
    # Variance
    ax_std.imshow(x_var[0, :][0], cmap="binary", vmin=0)

    return fig


def show_pseudo_inputs(ax, model):
    if (
        isinstance(model, V3AE)
        and getattr(model, "ood_z_generation_method", None) is not None
    ):
        # mult = getattr(model, "kde_bandwidth_multiplier", 10)
        # [n_mc_samples, BS, *self.latent_dims]
        z_out = next(iter(model.pig_dl))[0]
        ax.plot(
            z_out[:, 0],
            z_out[:, 1],
            "o",
            markersize=3.5,
            markerfacecolor=(228 / 255, 37 / 255, 101 / 255, 0.9),
            markeredgewidth=1.2,
            markeredgecolor=(*colours_rgb["white"], 0.5),
            label="PI",
        )
    return ax


def plot_geodesic_interpolation(
    model: Union[VanillaVAEm, V3AEm],
    z_start: torch.Tensor,
    z_end: torch.Tensor,
    N_interpolation: int,
) -> Tuple[BasicCurve, BasicCurve]:
    # Euclidean
    C_euclidean = DiscreteCurve(z_start, z_end)
    C_euclidean.plot()
    # Geodesic
    C_geodesic, success = model.connecting_geodesic(z_start, z_end)
    C_geodesic.plot()

    # 3 rows: mean, var, samples
    def plot_interpolation(
        model: Union[VanillaVAEm, V3AEm],
        curve: BasicCurve,
        N_interpolation: int,
        title: str,
    ):
        # Interpolants along the trajectory
        t_interpolation = torch.linspace(0, 1, N_interpolation)
        with torch.no_grad():
            interpolants = curve(t_interpolation)[None, :]

        # Reconstruction
        x_hat, x_mu, x_var = decode_from_latent(model, interpolants)

        fig = plt.figure()
        gs = fig.add_gridspec(
            3,
            N_interpolation,
            width_ratios=[1] * N_interpolation,
            height_ratios=[1] * 3,
        )
        gs.update(hspace=0.5, wspace=0.001)
        for i in range(N_interpolation):
            ax_mean, ax_var, ax_samples = (
                plt.subplot(gs[0, i]),
                plt.subplot(gs[1, i]),
                plt.subplot(gs[2, i]),
            )
            ax_mean.imshow(x_mu[i, 0, :], cmap="binary", vmin=0, vmax=1)
            ax_mean = disable_ticks(ax_mean)
            ax_var.imshow(x_var[i, 0, :], cmap="binary")
            ax_var = disable_ticks(ax_var)
            ax_samples.imshow(x_hat[i, 0, :], cmap="binary", vmin=0, vmax=1)
            ax_samples = disable_ticks(ax_samples)
        fig.suptitle(title.capitalize())
        return

    plot_interpolation(model, C_euclidean, N_interpolation, "euclidean")
    plot_interpolation(model, C_geodesic, N_interpolation, "geodesic")

    return C_euclidean, C_geodesic


def show_violin_plot_kl(model: V3AE, q_λ_z: D.Gamma, p_λ: D.Gamma, z: torch.Tensor):
    kl_lbd = model.kl(q_λ_z, p_λ).mean(dim=0)
    # [BS]
    kl_divergence_lbd_ood = model.ood_kl(p_λ, z)
    # If no ood
    if kl_divergence_lbd_ood.shape != kl_lbd.shape:
        kls = kl_lbd[None, :]
    else:
        kls = torch.cat((kl_lbd[None, :], kl_divergence_lbd_ood[None, :]), dim=0)
    fig, ax = plt.subplots()
    ax.violinplot(kls, showmeans=True)
    plt.show()


def show_per_pixel_uncertainty_kl(
    model: V3AE, z_star: Union[None, torch.Tensor], extent: Union[int, float]
):
    z_star = torch.Tensor([[-3.5, -3.5]]) if z_star is None else z_star
    z_star = z_star[None, :]
    _, _, α_z, β_z = model.parametrise_z(z_star)
    _, q_λ_z, p_λ = model.sample_precision(α_z, β_z)
    q_α, q_β = q_λ_z.base_dist.concentration.flatten(), q_λ_z.base_dist.rate.flatten()
    p_α, p_β = p_λ.base_dist.concentration.flatten(), p_λ.base_dist.rate.flatten()
    q_λ_z = D.Gamma(q_α, q_β)
    p_λ = D.Gamma(p_α, p_β)
    kl = model.kl(q_λ_z, p_λ)
    var = β_z / (α_z - 1)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    extent = 4
    ax1.imshow(
        var.reshape((28, 28)),
        extent=(-extent, extent, -extent, extent),
        vmax=np.percentile(var.flatten(), 75),
        # vmin=np.percentile(var.flatten(), 2),
        # cmap=cmap,
    )
    cax2 = ax2.imshow(
        kl.reshape((28, 28)),
        extent=(-extent, extent, -extent, extent),
        # vmax=np.percentile(kl.flatten(), 75),
        # vmin=np.percentile(var.flatten(), 2),
        # cmap=cmap,
    )
    ax2.title.set_text(f"Avg kl: {kl.mean():.2f}\nSum kl: {kl.sum():.2f}")
    fig.colorbar(cax2)
    plt.show()


def show_kl_imshow(
    model: V3AE,
    z: torch.Tensor,
    z_mesh: torch.Tensor,
    decoder_α_z_mesh: torch.Tensor,
    decoder_β_z_mesh: torch.Tensor,
    x_mesh_shape: Tuple[int, int],
    extent: Union[int, float],
    z_star: Union[None, torch.Tensor],
):
    _bs = 500
    N = decoder_α_z_mesh.shape[1]
    kl = torch.empty((N))
    for i in range(N // _bs):
        idx_low, idx_high = i * _bs, (i + 1) * _bs
        _decoder_α_z = decoder_α_z_mesh[0][idx_low:idx_high][None, :]
        _decoder_β_z = decoder_β_z_mesh[0][idx_low:idx_high][None, :]
        _z = z_mesh[idx_low:idx_high][None, :]
        _, _q_λ_z, _p_λ = model.sample_precision(_decoder_α_z, _decoder_β_z, _z)
        # [BS]
        _kl = model.kl(_q_λ_z, _p_λ).mean(dim=0)
        kl[idx_low:idx_high] = _kl

    kl = kl.reshape(x_mesh_shape)
    fig, ax = plt.subplots()
    cax = ax.imshow(
        kl,
        extent=(-extent, extent, -extent, extent),
        vmin=0,
        # vmax=15,
    )
    z_out = next(iter(model.pig_dl))[0]
    ax.plot(
        z_out[:, 0],
        z_out[:, 1],
        ".",
    )
    ax.plot(
        z[:, 0],
        z[:, 1],
        ".",
    )
    if isinstance(z_star, torch.Tensor):
        z_star = z_star.flatten()
        ax.plot(z_star[0], z_star[1], "o", markersize=6, color=colours["red"])
    ax.set_xlim([-extent, extent])
    ax.set_ylim([-extent, extent])
    fig.colorbar(cax)
    plt.show()


def plot_kl(
    model: V3AE,
    z: torch.Tensor,
    z_mesh: torch.Tensor,
    z_star: torch.Tensor,
    x_mesh_shape: Tuple[int, int],
    extent: Union[int, float],
    # %
    # Change here for display
    show_imshow: bool = False,
    show_per_pixel: bool = False,
    show_violin: bool = False,
):
    # %
    if show_imshow:
        _, _, α_z_mesh, β_z_mesh = model.parametrise_z(z_mesh[None, :])
        show_kl_imshow(
            model, z, z_mesh, α_z_mesh, β_z_mesh, x_mesh_shape, extent, z_star
        )
    # %
    if show_per_pixel:
        show_per_pixel_uncertainty_kl(model, z_star, extent)
    # %
    if show_violin:
        _, _, α_z, β_z = model.parametrise_z(z[None, :])
        _, q_λ_z, p_λ = model.sample_precision(α_z, β_z)
        show_violin_plot_kl(model, q_λ_z, p_λ, z[None, :])


# ==================================================================================
def show_reconstruction_grid(
    model: Union[VanillaVAE, V3AE],
    size: int = 20,
):
    extent = 4
    x_mesh = torch.linspace(-extent, extent, size)
    # Imshow starts by default from upper left
    y_mesh = torch.linspace(extent, -extent, size)
    x_mesh, y_mesh = torch.meshgrid(x_mesh, y_mesh)
    x_mesh, y_mesh = x_mesh.transpose(0, 1), y_mesh.transpose(0, 1)
    z_mesh = torch.cat((x_mesh.flatten()[:, None], y_mesh.flatten()[:, None]), dim=1)

    with torch.no_grad():
        if isinstance(model, VanillaVAE):
            μ_z, std_z = model.decoder_μ(z_mesh), model.decoder_std(z_mesh)
            x_hat, p_x = model.sample_generative(μ_z, std_z)
        elif isinstance(model, V3AE):
            _, μ_z, α_z, β_z = model.parametrise_z(z_mesh[None, :])
            x_hat, p_x = model.sample_generative(μ_z, α_z, β_z)
        x_mean = p_x.mean

    # reshape
    x_hat, x_mean = x_hat.reshape(-1, 1, 28, 28), x_mean.reshape(-1, 1, 28, 28)
    x_hat, x_mean = x_hat.clamp(0, 1), x_mean.clamp(0, 1)

    # grid
    x_hat, x_mean = (
        make_grid(x_hat, nrow=size),
        make_grid(x_mean, nrow=size),
    )

    # to grayscale
    tf_to_gs = transforms.Compose(
        [transforms.ToPILImage(), transforms.Grayscale(), transforms.ToTensor()]
    )
    x_hat, x_mean = tf_to_gs(x_hat)[0], tf_to_gs(x_mean)[0]

    # plot
    fig_hat, ax_hat = plt.subplots()
    ax_hat.imshow(x_hat, cmap="binary", vmin=0, vmax=1)

    fig_mean, ax_mean = plt.subplots()
    ax_mean.imshow(x_mean, cmap="binary", vmin=0)

    return fig_hat, fig_mean


def show_2d_latent_space(
    model,
    x,
    y,
    title="TITLE",
    show_geodesic=True,
    show_kl=True,
    show_pi=True,
    z_star: Union[None, torch.Tensor] = None,
):
    digits = torch.unique(y)
    with torch.no_grad():
        if isinstance(model, VanillaVAE):
            _, _, z, _, _ = model._run_step(x)
        elif isinstance(model, V3AE):
            _, _, _, q_λ_z, p_λ, z, q_z_x, _ = model._run_step(x)
            # %
            # keep only a single z sample
            z = z[0]

    # Show imshow for variance -> inspiration from aleatoric_epistemic_split
    extent, N_mesh = 5, 100
    x_mesh = torch.linspace(-extent, extent, 100)
    y_mesh = torch.linspace(-extent, extent, 100)
    x_mesh, y_mesh = torch.meshgrid(x_mesh, y_mesh)
    z_mesh = torch.cat((x_mesh.flatten()[:, None], y_mesh.flatten()[:, None]), dim=1)
    with torch.no_grad():
        if isinstance(model, VanillaVAE):
            var = model.decoder_std(z_mesh)
        if isinstance(model, V3AE):
            # [1, BS(meshgrid # positions), input_size]
            _, _, decoder_α_z, decoder_β_z = model.parametrise_z(z_mesh[None, :])
            var = decoder_β_z[0] / (decoder_α_z[0] - 1)
            # %
            z_tst = torch.Tensor([1, 1]).reshape(1, 1, 2)
            _, _, alpha_tst, beta_tst = model.parametrise_z(z_tst)
            # %
            if show_kl:
                plot_kl(model, z, z_mesh, z_star, x_mesh.shape, extent)

    # Accumulated gradient over all output cf nicki and martin
    var = torch.mean(var, dim=1)
    # reshape to x_shape
    var = var.reshape(*x_mesh.shape)

    fig, ax = plt.subplots()
    cmap = sns.color_palette("rocket", as_cmap=True)
    varimshw = ax.imshow(
        var,
        extent=(-extent, extent, -extent, extent),
        vmax=np.percentile(var.flatten(), 75),
        # vmin=np.percentile(var.flatten(), 2),
        cmap=cmap,
    )

    # Show two digits separately
    # cla = ["Pullover", "Sandal"]
    for i, d in enumerate(digits):
        ax.plot(
            z[:, 0][y == d],
            z[:, 1][y == d],
            "o",
            markersize=3.5,
            markerfacecolor=(*colour_digits[i], 0.95),
            markeredgewidth=1.2,
            markeredgecolor=(*colours_rgb["white"], 0.5),
            label=f"Digit {d}",
        )

    if isinstance(z_star, torch.Tensor):
        z_star = z_star.flatten()
        ax.plot(z_star[0], z_star[1], "o", markersize=6, color="#0AEFFF")

    # Pseudo-inputs
    if show_pi:
        show_pseudo_inputs(ax, model)

    # Geodesic
    if show_geodesic and isinstance(model, EmbeddedManifold):
        z_start = torch.Tensor([-0.04, -1.13])
        z_end = torch.Tensor([0, 1.9])
        N_interpolation = 10
        C_euclidean, C_geodesic = plot_geodesic_interpolation(
            model, z_start, z_end, N_interpolation
        )

    # Misc
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([-extent, extent])
    ax.set_ylim([-extent, extent])
    cb = fig.colorbar(varimshw, ax=ax)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(20)
    ax.legend(
        fancybox=True,
        shadow=False,
        # ncol=legend_ncols,
        loc="upper left",
        bbox_to_anchor=(-0.38, 1),
    )
    ax.set_title(title, fontsize=30)

    return fig
