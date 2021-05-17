import math
import matplotlib.pyplot as plt
import numpy as np
import torch

from pytorch_lightning import seed_everything
from sggm.analysis.experiment_log import ExperimentLog
from sggm.data.toy_symmetrical import ToySymmetricalDataModule
from sggm.regression_model import VariationalRegressor
from sggm.styles_ import colours, colours_rgb

""" Plot to demonstrate the effect of PIs placement on the learned uncertainty for toy. """

# Plot data and params
def get_plot_dataset(N: int = 5000):
    data_range_plot = [0, 10]
    x = torch.linspace(*data_range_plot, steps=N)[:, None]

    return (
        x,
        ToySymmetricalDataModule.data_mean(x),
        ToySymmetricalDataModule.data_std(x),
    )


def plot_uncertainty(ax, data_range):
    alpha_lines = 0.95
    x = torch.linspace(*data_range, steps=1000)

    # True
    ax.plot(
        x_plot[training_mask],
        y_std_plot[training_mask],
        color=(*colours_rgb["black"], alpha_lines),
        linewidth=1.2,
    )
    x_displayed = 10 * torch.rand((50,))
    ax.plot(
        x_displayed,
        1e-2 + torch.zeros_like(x_displayed),
        "o",
        markersize=5,
        markerfacecolor=(*colours_rgb["black"], 0.8),
        markeredgewidth=1,
        markeredgecolor=(*colours_rgb["black"], 0.2),
        zorder=10,
    )

    # Prior
    ax.plot(
        x,
        prior_level * torch.ones_like(x),
        color="grey",
        linewidth=1.1,
        linestyle="dotted",
    )

    # No PIs
    ax.plot(
        x,
        model_no_pis.predictive_std(x.reshape(-1, 1)),
        color=(*colours_rgb["primaryRed"], alpha_lines),
        linewidth=1.4,
    )

    # Close PIs
    ax.plot(
        x,
        model_close_pis.predictive_std(x.reshape(-1, 1)),
        color=(*colours_rgb["green"], alpha_lines),
        linewidth=1.4,
    )

    # Far PIs
    ax.plot(
        x,
        model_far_pis.predictive_std(x.reshape(-1, 1)),
        color=(*colours_rgb["navyBlue"], alpha_lines),
        linewidth=1.4,
    )

    # ax.set_yticks([])
    # ax.set_xticks([])
    ax.set_facecolor("#F7F8F6")
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlabel(r"\textnormal{x}", fontsize=15)
    ax.set_ylabel(r"$\sigma(\textnormal{x})$", fontsize=15)
    ax.set_xlim(data_range)
    ax.set_ylim([-0.2, 2.2])
    ax.grid(True, color="white")

    return ax


def save_show(suffix: str) -> None:
    plt.tight_layout()
    plt.savefig(
        f"_paper_nips_plots/pi_placement_{suffix}.png",
        bbox_inches="tight",
    )
    plt.savefig(
        f"_paper_nips_plots/pi_placement_{suffix}.svg",
        bbox_inches="tight",
    )

    plt.show()


def generate_pis(range_left, range_right, N) -> torch.Tensor:
    left = torch.FloatTensor(int(N / 2), 1).uniform_(*range_left)
    right = torch.FloatTensor(int(N / 2), 1).uniform_(*range_right)
    return torch.cat((left, right))


if __name__ == "__main__":
    seed_everything(14)
    experiment_name = "toy_symmetrical"
    model_name = "variational_regressor"
    save_dir = "paper_logs"

    # Load the models
    model_no_pis: VariationalRegressor = ExperimentLog(
        experiment_name, "svv", model_name=model_name, save_dir=save_dir
    ).best_version.model
    model_close_pis: VariationalRegressor = ExperimentLog(
        experiment_name, "uniform_close", model_name=model_name, save_dir=save_dir
    ).best_version.model
    model_far_pis: VariationalRegressor = ExperimentLog(
        experiment_name, "uniform_far", model_name=model_name, save_dir=save_dir
    ).best_version.model

    prior_level = math.sqrt(model_close_pis.prior_β / (model_close_pis.prior_α - 1))

    x_plot, y_plot, y_std_plot = get_plot_dataset()
    data_range_training = [0, 10]
    training_mask = (data_range_training[0] <= x_plot) * (
        x_plot <= data_range_training[1]
    )

    fig_size = (5.8, 2)
    # Close
    fig_close, ax_close = plt.subplots(figsize=fig_size)
    ax_close = plot_uncertainty(ax_close, [-10, 20])
    # close pseudo_inputs
    pi_close = generate_pis([-3, 0.5], [10.2, 13], 20)
    ax_close.plot(
        pi_close,
        1e-2 + torch.zeros_like(pi_close),
        "D",
        markersize=5,
        markerfacecolor=(*colours_rgb["green"], 0.8),
        markeredgewidth=1,
        markeredgecolor=(*colours_rgb["green"], 0.2),
        zorder=10,
    )
    save_show("close")
    # Far
    fig_far, ax_far = plt.subplots(figsize=fig_size)
    ax_far = plot_uncertainty(ax_far, [-120, 200])
    pi_far = generate_pis([-120, -100], [170, 200], 20)
    ax_far.plot(
        pi_far,
        1e-2 + torch.zeros_like(pi_far),
        "D",
        markersize=5,
        markerfacecolor=(*colours_rgb["navyBlue"], 0.8),
        markeredgewidth=1,
        markeredgecolor=(*colours_rgb["navyBlue"], 0.2),
        zorder=10,
    )
    save_show("far")
