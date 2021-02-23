import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from torch import no_grad

from sggm.vae_model import V3AE, VanillaVAE
from sggm.vae_model_helper import batch_flatten, batch_reshape


def disable_ticks(ax: matplotlib.axes.Axes) -> matplotlib.axes.Axes:
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax.tick_params(axis="y", which="both", left=False, labelleft=False)
    return ax


def input_to_latent(model, x):
    x = batch_flatten(x)
    μ_x = model.encoder_μ(x)
    std_x = model.encoder_std(x)
    z, _, _ = model.sample_latent(μ_x, std_x)
    return z


def plot_comparison(n_display, x_og, p_x, input_dims):
    fig = plt.figure(figsize=(6, 6))
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


def get_interpolation_digits(
    dm, experiment_name, target_digits: [1, 3], target_digits_idx: [0, 0]
):
    """
    Organise class elements per class value and samples elements from it
    """
    digits = [[] for _ in range(10)]
    for img_batch, label_batch in dm.test_dataloader():
        for i in range(img_batch.size(0)):
            digits[label_batch[i]].append(img_batch[i : i + 1])
        # Only fetch 100 class elements
        if sum(len(d) for d in digits) >= 100:
            break

    interpolation_digits = [
        digits[target_digits[0]][target_digits_idx[0]],
        digits[target_digits[1]][target_digits_idx[1]],
    ]
    return interpolation_digits


def latent_to_moments(model, z):
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
    return batch_reshape(p_x_z.mean, model.input_dims), batch_reshape(
        p_x_z.variance, model.input_dims
    )


def interpolation(tau, model, img1, img2):

    with no_grad():
        # latent vector of first image
        latent_1 = input_to_latent(model, img1)

        # latent vector of second image
        latent_2 = input_to_latent(model, img2)

        # Interpolation 2D, choose arbitrary position of interpolants
        # through center
        # latent_1 = torch.Tensor([[[0.0, 1.0]]])
        # latent_2 = torch.Tensor([[[0.0, -1.0]]])
        # through circle
        # latent_1 = torch.Tensor([[[1.5, 1.0]]])
        # latent_2 = torch.Tensor([[[1, -1.0]]])

        # interpolation of the two latent vectors
        inter_latent = tau * latent_1 + (1 - tau) * latent_2

        # reconstruct interpolated image
        inter_image, inter_variance = latent_to_moments(model, inter_latent)
        inter_image = inter_image.numpy()
        inter_variance = inter_variance.numpy()

    return inter_image, inter_variance


def plot_interpolation(model, img1, img2):
    n_img = 10
    tau_range = np.flip(np.linspace(0, 1, n_img))

    fig = plt.figure()  # constrained_layout=True
    gs = fig.add_gridspec(
        int(n_img / 5), 5, width_ratios=[1] * 5, height_ratios=[1] * int(n_img / 5)
    )
    gs.update(hspace=0.5, wspace=0.001)

    for i, tau in enumerate(tau_range):
        row, col = i // 5, i % 5

        interpolated_img, interpolated_var = interpolation(
            float(tau), model, img1, img2
        )

        ax = plt.subplot(gs[row, col])
        ax = disable_ticks(ax)
        ax.imshow(interpolated_img[0][0], cmap="binary")
        ax.set_title(r"$\rho\,=\,$" + f"{1 - round(tau, 1):.1f}")

    return fig
