import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch

from torch import no_grad

from sggm.vae_model import V3AE, VanillaVAE
from sggm.vae_model_helper import batch_reshape
from sggm.styles_ import colours, colours_rgb

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
    print(x_var.min(), x_var.max(), x_var.mean(), x_var.std())
    print(x_var.shape)

    fig, (ax_reconstruct, ax_mean, ax_std) = plt.subplots(3, 1)
    # Reconstruction
    ax_reconstruct.imshow(x_hat[0, :][0], cmap="binary", vmin=0, vmax=1)
    # Mean
    ax_mean.imshow(x_mu[0, :][0], cmap="binary", vmin=0, vmax=1)
    # Variance
    ax_std.imshow(x_var[0, :][0], cmap="binary", vmin=0, vmax=50)

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


def show_2d_latent_space(model, x, y, title="TITLE", show_pi=True, z_star=None):
    digits = torch.unique(y)
    with torch.no_grad():
        if isinstance(model, VanillaVAE):
            _, _, z, _, _ = model._run_step(x)
        elif isinstance(model, V3AE):
            _, _, _, _, _, z, q_z_x, _ = model._run_step(x)
            # keep only a single z sample
            z = z[0]

    fig, ax = plt.subplots()
    # Show imshow for variance -> inspiration from aleatoric_epistemic_split
    extent = 10
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
    # Accumulated gradient over all output cf nicki and martin
    var = torch.mean(var, dim=1)
    # reshape to x_shape
    var = var.reshape(*x_mesh.shape)

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
