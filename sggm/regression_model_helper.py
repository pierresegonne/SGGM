import numpy as np
import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader, TensorDataset


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


def sigmoid(x, mu=0, sigma=1):
    if isinstance(x, np.ndarray):
        x = torch.Tensor(x)
    x = mu + (x * sigma)
    return 1 / (1 + torch.exp(-x))


def mean_shift_pig_dl(
    dm: pl.LightningDataModule,
    batch_size: int,
    N_hat: int = 100,
    max_iters: int = 20,
    t_box: float = 0.75,
    sigma=0.1,
) -> DataLoader:
    t_box = torch.Tensor([t_box])

    x = next(iter(dm.train_dataloader()))[0]
    for idx, batch in enumerate(iter(dm.train_dataloader())):
        if idx == 0:
            continue
        x = torch.cat((x, batch[0]))

    # With replacement to allow for N_hat > N
    idx = torch.randint(x.shape[0], (N_hat,))
    # Without replacement
    # idx = torch.randperm(x.shape[0])[:N_hat]
    # [n_pig, D]
    x_start = x[idx]

    # Sample locally around each point, [n_pig, D]
    x_hat = x_start + sigma * torch.randn_like(x_start)

    for _ in range(max_iters):
        # [N, n_pig]
        dst = sqr_dist(x, x_hat)
        dst = torch.where(
            dst <= -2 * torch.log(t_box), torch.ones_like(dst), torch.zeros_like(dst)
        )
        # Replace nan by 0
        dst[dst != dst] = 0

        # [n_pig, D]
        # Centroid for all estimates
        sum_dst = torch.sum(dst, dim=0).reshape((-1, 1))
        # Threshold for stopping updates
        out_dst = sum_dst < 1e-5
        sum_dst = torch.where(out_dst, torch.ones_like(sum_dst), sum_dst)

        mu = (torch.transpose(dst, 0, 1) @ x) / sum_dst
        delta = (mu - x_hat) * sigmoid(sum_dst, mu=-5, sigma=1 / 25)

        x_hat = torch.where(out_dst, x_hat, x_hat - delta)

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

    dl = DataLoader(TensorDataset(x_hat), batch_size=batch_size, shuffle=True)

    return dl
