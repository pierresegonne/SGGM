import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.distributions as D

from torchvision.utils import save_image, make_grid

from sggm.vae_model import VanillaVAE, V3AE, V3AEm
from sggm.vae_model_helper import batch_reshape
from sggm.styles_ import colours, colours_rgb
from sggm.types_ import Tuple, Union

colour_digits = [
    (1 / 255, 133 / 255, 90 / 255),
    (200 / 255, 154 / 255, 1 / 255),
]


def show_reconstruction_arbitrary_latent(
    model: pl.LightningModule, z_star: torch.Tensor
) -> plt.Figure:
    _, μ_z, α_z, β_z = model.parametrise_z(z_star)
    x_hat, p_x = model.sample_generative(μ_z, α_z, β_z)

    x_hat = batch_reshape(x_hat, model.input_dims)
    x_mu = batch_reshape(p_x.mean, model.input_dims)
    x_var = batch_reshape(p_x.variance, model.input_dims)

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
            markerfacecolor=(*colours_rgb["purple"], 0.9),
            markeredgewidth=1.2,
            markeredgecolor=(*colours_rgb["white"], 0.5),
            label="PI",
        )
    return ax


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
    decoder_α_z: torch.Tensor,
    decoder_β_z: torch.Tensor,
    x_mesh_shape: Tuple[int, int],
    extent: Union[int, float],
    z_star: Union[None, torch.Tensor],
):
    _bs = 500
    N = decoder_α_z.shape[1]
    kl = torch.empty((N))
    for i in range(N // _bs):
        idx_low, idx_high = i * _bs, (i + 1) * _bs
        _decoder_α_z = decoder_α_z[0][idx_low:idx_high][None, :]
        _decoder_β_z = decoder_β_z[0][idx_low:idx_high][None, :]
        _, _q_λ_z, _p_λ = model.sample_precision(_decoder_α_z, _decoder_β_z)
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
    show_imshow: bool = True,
    show_per_pixel: bool = False,
    show_violin: bool = False,
):
    # %
    if show_imshow:
        _, _, α_z, β_z = model.parametrise_z(z_mesh[None, :])
        show_kl_imshow(model, z, α_z, β_z, x_mesh_shape, extent, z_star)
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
    size: int = 50,
):
    extent = 4
    x_mesh = torch.linspace(-extent, extent, size)
    y_mesh = torch.linspace(-extent, extent, size)
    x_mesh, y_mesh = torch.meshgrid(x_mesh, y_mesh)
    z_mesh = torch.cat((x_mesh.flatten()[:, None], y_mesh.flatten()[:, None]), dim=1)[
        None, :
    ]

    with torch.no_grad():
        _, μ_z, α_z, β_z = model.parametrise_z(z_mesh)
        x_hat, p_x = model.sample_generative(μ_z, α_z, β_z)
        x_mean = p_x.mean

    # reshape
    x_hat, x_mean = x_hat.reshape(-1, 1, 28, 28), x_mean.reshape(-1, 1, 28, 28)

    # grid
    x_hat, x_mean = (
        make_grid(x_hat, nrow=size).permute(1, 2, 0),
        make_grid(x_mean, nrow=size).permute(1, 2, 0),
    )
    fig_hat, ax_hat = plt.subplots()
    ax_hat.imshow(x_hat)

    fig_mean, ax_mean = plt.subplots()
    ax_mean.imshow(x_mean)

    return fig_hat, fig_mean


def show_2d_latent_space(
    model,
    x,
    y,
    title="TITLE",
    show_geodesic=False,
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
    extent = 5
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
        ax.plot(z_star[0], z_star[1], "o", markersize=6, color=colours["red"])

    # Pseudo-inputs
    legend_ncols = 2
    if show_pi:
        show_pseudo_inputs(ax, model)
        legend_ncols = 3

    # Geodesic
    if show_geodesic and isinstance(model, V3AEm):
        z1 = torch.Tensor([-0.58, -1.13])
        z2 = torch.Tensor([0.4, 1.1])
        C, success = model.connecting_geodesic(z1, z2)
        C.plot()

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
        ncol=legend_ncols,
        bbox_to_anchor=(0.89, -0.15),
    )
    ax.set_title(title, fontsize=30)

    return fig
