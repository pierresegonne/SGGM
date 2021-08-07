import numpy as np
import torch
import torch.distributions as D
import pytorch_lightning as pl

from sklearn.mixture import BayesianGaussianMixture
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple

from sggm.definitions import OOD_X_GENERATION_AVAILABLE_METHODS
from sggm.model_helper import density_gradient_descent


def check_ood_x_generation_method(method: str) -> str:
    if method is None:
        return method
    assert (
        method in OOD_X_GENERATION_AVAILABLE_METHODS
    ), f"""Method for x ood generation '{method}' is invalid.
    Must either be None or in {OOD_X_GENERATION_AVAILABLE_METHODS}"""
    return method


def check_mixture_ratio(r: float) -> float:
    assert (r >= 0) & (r <= 1), "Invalid ratio"
    return r


def generate_noise_for_model_test(x: torch.Tensor) -> torch.Tensor:
    """Generates noisy inputs to test the model out of distribution

    Args:
        x (torch.Tensor): testing inputs

    Returns:
        torch.Tensor: Noisy inputs
    """
    hypercube_min, _ = torch.min(x, dim=0)
    hypercube_max, _ = torch.max(x, dim=0)

    data_std = x.std(dim=0)
    data_mean = x.mean(dim=0)

    noise = torch.rand(x.shape).type_as(x) - 0.5
    noise *= (hypercube_max - hypercube_min) * 2 * data_std
    noise += data_mean

    return noise


def sqr_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # x [N, D] - y [M, D] -> (x-y)**2 [N, M]
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
    d2 = x_norm + y_norm - 2.0 * torch.mm(x, y.transpose(0, 1))
    return d2.clamp(min=0.0)  # NxM


def silverman_bandwidth(x: torch.Tensor) -> torch.Tensor:
    _x = x.cpu().numpy()
    sigma = _x.std(axis=0)
    iqr = np.subtract(*np.percentile(_x, [75, 25], axis=0))
    n = _x.shape[0]
    return torch.Tensor(
        [(0.9 * np.minimum(sigma, iqr / 1.34) * (n ** (-1 / 5))).mean()]
    ).type_as(x)


def unpack_dataloader(dl: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    # Unpack x
    x, y = next(iter(dl))
    for idx, batch in enumerate(iter(dl)):
        if idx == 0:
            continue
        x = torch.cat((x, batch[0]))
        y = torch.cat((y, batch[1]))

    return x, y


def gaussian_noise_pig_dl(
    dm: pl.LightningDataModule,
    batch_size: int,
    N_hat_multiplier: float = 1,
    sigma_multiplier: float = 1,
) -> DataLoader:
    #  * 3  # 3 is Arbitrary
    # return x + noise_std * torch.randn_like(x)
    x, _ = unpack_dataloader(dm.train_dataloader())
    N = x.shape[0]
    noise_std = torch.std(x)

    # Does not exactly respect N_hat but easier to do that way
    if N_hat_multiplier >= 1:
        n_repeats = int(N_hat_multiplier)
        x = x.repeat(1, n_repeats).view(-1, x.shape[1])
    else:
        N_hat = int(N * N_hat_multiplier)
        idx = torch.randperm(x.shape[0])[:N_hat]
        x = x[idx]

    x_hat = x + noise_std * torch.randn_like(x)

    dl = DataLoader(TensorDataset(x_hat), batch_size=batch_size, shuffle=True)
    return dl


def kde_pig_dl(
    dm: pl.LightningDataModule,
    batch_size: int,
    N_hat_multiplier: float = 1,
    gd_n_steps: int = 5,
    gd_lr: float = 4e-1,
    gd_threshold: float = 0.005,
) -> DataLoader:

    # Spherical = each component has single variance.
    bgm = BayesianGaussianMixture(
        n_components=batch_size,
        covariance_type="spherical",
        warm_start=True,
    )

    x_hat = torch.Tensor()
    for idx, batch in enumerate(iter(dm.train_dataloader())):
        x, _ = batch
        device = x.device
        x = x.detach().cpu().numpy()
        # Last batch might have less elements than origin n_components
        if x.shape[0] < bgm.n_components:
            bgm = BayesianGaussianMixture(
                n_components=x.shape[0],
                covariance_type="spherical",
            )
        # Estimate KDE
        bgm.fit(x)
        # [N_components, 1], [N_components, N_features], [N_components, 1]
        weights, means, variances = (
            torch.Tensor(bgm.weights_).to(device),
            torch.Tensor(bgm.means_).to(device),
            torch.Tensor(bgm.covariances_).to(device),
        )
        filter_weights_idx = weights >= 1e-5
        weights, means, variances = (
            weights[filter_weights_idx],
            means[filter_weights_idx],
            variances[filter_weights_idx][:, None],
        )
        n_selected_components = weights.shape[0]
        p_x = D.Independent(D.Normal(means, torch.sqrt(variances)), 1)
        mix = D.Categorical(weights)
        p_x = D.MixtureSameFamily(mix, p_x)
        # Sample according to multiplier
        x_start = p_x.sample(
            (
                n_selected_components
                * ((batch_size // n_selected_components) + 1)
                * N_hat_multiplier,
            )
        ).reshape(-1, x.shape[1])
        # Use GD
        _x_hat = density_gradient_descent(
            p_x,
            x_start,
            {"N_steps": gd_n_steps, "lr": gd_lr, "threshold": gd_threshold},
        )
        # Ensure same device
        if x_hat.device != device:
            x_hat = x_hat.to(device)
        x_hat = torch.cat((x_hat, _x_hat.detach()))

    dl = DataLoader(TensorDataset(x_hat), batch_size=batch_size, shuffle=True)
    return dl


def mean_shift_pig_dl(
    dm: pl.LightningDataModule,
    batch_size: int,
    N_hat_multiplier: float = 1 / 20,
    max_iters: int = 20,
    h: float = None,
    h_factor: float = 1,
    sigma: float = None,
    sigma_factor: float = 1,
    kernel: str = "tophat",
    τ: float = 1e-5,
) -> DataLoader:
    x, _ = unpack_dataloader(dm.train_dataloader())

    if h is None:
        h = silverman_bandwidth(x) * h_factor
    if sigma is None:
        # Note multiplier is arbitrary
        sigma = torch.std(x) * sigma_factor

    N = x.shape[0]
    N_hat = int(N_hat_multiplier * N)
    if N_hat > N:
        # With replacement to allow for N_hat > N
        idx = torch.randint(x.shape[0], (N_hat,))
    else:
        # Without replacement
        idx = torch.randperm(x.shape[0])[:N_hat]
    # [n_pig, D]
    x_start = x[idx]

    # Sample locally around each point, [n_pig, D]
    x_hat = x_start + sigma * torch.randn_like(x_start)

    for _ in range(max_iters):
        # [N, n_pig]
        dst = sqr_dist(x, x_hat)
        if kernel == "tophat":
            kde = torch.where(dst <= h, torch.ones_like(dst), torch.zeros_like(dst))
        else:
            raise NotImplementedError(f"Kernel {kernel} invalid")
        # Replace nan by 0
        kde[kde != kde] = 0

        # Threshold for stopping updates
        out_kde = (torch.max(kde, dim=0)[0] < τ)[:, None]

        # [n_pig, 1]
        sum_kde = torch.sum(kde, dim=0).reshape((-1, 1))
        sum_kde = torch.where(out_kde, torch.ones_like(sum_kde), sum_kde)

        # Centroid for all estimates
        mu = (torch.transpose(kde, 0, 1) @ x) / sum_kde
        # Step size
        delta = mu - x_hat

        x_hat = torch.where(out_kde, x_hat, x_hat - delta)

    # import matplotlib.pyplot as plt
    # from sklearn.neighbors import KernelDensity

    # kde = KernelDensity(kernel="gaussian", bandwidth=0.2).fit(x_hat)
    # x_plot = np.linspace(-10, 20, 1000).reshape(-1, 1)
    # density = np.exp(kde.score_samples(x_plot))

    # fig, ax = plt.subplots()
    # ax.plot(x, torch.zeros_like(x), "o")
    # ax.plot(x_hat, torch.zeros_like(x_hat), "o")
    # ax.plot(x_plot, density, color="black")

    # plt.show()
    # exit()

    dl = DataLoader(TensorDataset(x_hat), batch_size=batch_size, shuffle=True)

    return dl
