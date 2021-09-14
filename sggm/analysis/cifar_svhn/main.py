from typing import Tuple, Union

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from pytorch_lightning import seed_everything, LightningDataModule
import torch
import torch.distributions as D

from sggm.analysis.utils import disable_ticks
from sggm.data.cifar import CIFARDataModule
from sggm.data.svhn import SVHNDataModule
from sggm.definitions import CIFAR, SVHN
from sggm.vae_model_helper import batch_reshape


def get_dm(experiment_name: str, bs: int):
    if experiment_name == CIFAR:
        dm = CIFARDataModule(bs, 0)
    elif experiment_name == SVHN:
        dm = SVHNDataModule(bs, 0)
    dm.setup()
    return dm


def plot_comparison(
    n_display: int, x_og: torch.Tensor, p_x: D.Distribution, input_dims: Tuple[int]
) -> Figure:
    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(
        4, n_display, width_ratios=[1] * n_display, height_ratios=[1, 1, 1, 1]
    )
    gs.update(wspace=0, hspace=0)

    x_hat = batch_reshape(p_x.sample(), input_dims).clip(0, 1)
    x_mu = batch_reshape(p_x.mean, input_dims).clip(0, 1)
    x_var = batch_reshape(p_x.variance, input_dims).clip(0, 1)

    for n in range(n_display):
        for k in range(4):
            ax = plt.subplot(gs[k, n])
            ax = disable_ticks(ax)
            # Original
            # imshow accepts (W, H, C)
            if k == 0:
                ax.imshow(x_og[n, :].permute(1, 2, 0), vmin=0, vmax=1)
            # Mean
            elif k == 1:
                ax.imshow(x_mu[n, :].permute(1, 2, 0), vmin=0, vmax=1)
            # Variance
            elif k == 2:
                ax.imshow(x_var[n, :].permute(1, 2, 0))
            # Sample
            elif k == 3:
                ax.imshow(x_hat[n, :].permute(1, 2, 0), vmin=0, vmax=1)

    return fig


def plot(
    experiment_log,
    seed=False,
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
        dm = get_dm(experiment_name, bs)

    # Dataset
    test_dataset = next(iter(dm.val_dataloader()))
    x_test, y_test = test_dataset

    # Reconstruction
    best_model.eval()
    # %
    best_model.dm = dm
    with torch.no_grad():
        x_hat_test, p_x_test = best_model(x_test)

    fig = plot_comparison(4, x_test, p_x_test, best_model.input_dims)
    fig.savefig(f"{save_folder}/_main.png", dpi=300)
    fig.savefig(f"{save_folder}/_main.svg")
    plt.show()
