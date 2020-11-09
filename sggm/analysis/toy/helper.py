import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.distributions as tcd

from typing import Tuple
from scipy.stats import norm
from sggm.data.toy import ToyDataModule
from sggm.regression_model import (
    fit_prior,
    MARGINAL,
    POSTERIOR,
)
from sggm.styles_ import colours, colours_rgb

# ------------
# Plot data definition
# ------------
data_range_plot = [-4, 14]
data_range_training = [0, 10]


def get_plot_dataset(N: int = 1000) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.linspace(*data_range_plot, steps=N)[:, None]

    return x, ToyDataModule.data_mean(x), ToyDataModule.data_std(x)


plot_dataset = get_plot_dataset()


def get_colour_for_method(method):
    if method == MARGINAL:
        return colours["orange"]
    elif method == POSTERIOR:
        return colours["navyBlue"]


# ------------
# Plot methods
# ------------
def base_plot(
    data_ax,
    std_ax,
    data=plot_dataset,
    range_=data_range_training,
    plot_lims={
        "data_ax": [-25, 25],
        "std_ax": [0, 5],
    },
):

    # Unpack plot dataset
    x_plot, y_plot, y_std_plot = data
    y_plot_mstd = y_plot - 1.96 * y_std_plot
    y_plot_pstd = y_plot + 1.96 * y_std_plot

    # Plot limits
    plot_x_range = [torch.min(x_plot) - 1, torch.max(x_plot) + 1]

    # Plot
    data_ax.plot(x_plot, y_plot, color="black", linestyle="dashed", linewidth=1)
    data_ax.plot(x_plot, y_plot_mstd, color="black", linestyle="dotted", linewidth=0.5)
    data_ax.plot(x_plot, y_plot_pstd, color="black", linestyle="dotted", linewidth=0.5)
    data_ax.grid(True)
    data_ax.set_ylabel("y")
    data_ax.set_xlim(plot_x_range)
    data_ax.set_ylim(plot_lims["data_ax"])

    training_mask = (range_[0] <= x_plot) * (x_plot <= range_[1])
    std_ax.plot(
        x_plot[training_mask], y_std_plot[training_mask], color="black", linewidth=1
    )
    b_training_mask = x_plot <= range_[0]
    a_training_mask = x_plot >= range_[1]
    std_ax.plot(
        x_plot[b_training_mask],
        y_std_plot[b_training_mask],
        color="black",
        linestyle="dashdot",
        linewidth=1,
    )
    std_ax.plot(
        x_plot[a_training_mask],
        y_std_plot[a_training_mask],
        color="black",
        linestyle="dashdot",
        linewidth=1,
    )
    std_ax.grid(True)
    std_ax.set_xlabel("x")
    std_ax.set_ylabel(r"$\sigma(y|x)$")
    std_ax.set_xlim(plot_x_range)
    std_ax.set_ylim(plot_lims["std_ax"])

    return data_ax, std_ax


def training_points_plot(data_ax, training_dataset):
    x_train, y_train = training_dataset
    data_ax.plot(
        x_train,
        y_train,
        "o",
        markersize=3,
        markerfacecolor=(*colours_rgb["navyBlue"], 0.6),
        markeredgewidth=1,
        markeredgecolor=(*colours_rgb["navyBlue"], 0.1),
    )
    return data_ax


def best_model_plot(data_ax, model, method):
    x_plot = plot_dataset[0]
    colour = get_colour_for_method(method)
    best_mean = model.predictive_mean(x_plot, method).flatten()
    best_std = model.predictive_std(x_plot, method).flatten()

    data_ax.plot(x_plot, best_mean, "-", color=colour, alpha=0.55)
    data_ax.fill_between(
        x_plot.flatten(),
        best_mean + 1.96 * best_std,
        best_mean - 1.96 * best_std,
        facecolor=colour,
        alpha=0.3,
    )
    return data_ax


def mean_models_plot(std_ax, experiment_log, method):
    # Get df for all trials for the std
    colour = get_colour_for_method(method)
    std_df = get_std_trials_df(experiment_log.versions, method)
    sns.lineplot(
        x="x",
        y="std(y|x)",
        ci="sd",
        data=std_df,
        ax=std_ax,
        color=colour,
        alpha=0.55,
    )
    return std_ax


def get_std_trials_df(versions, method):
    with torch.no_grad():
        x_plot, _, _ = plot_dataset

        df = pd.DataFrame(columns=["x", "std(y|x)"])

        for v in versions:
            std = v.model.predictive_std(x_plot, method)
            std = std.numpy().reshape((-1, 1))

            new_trial_stds = np.concatenate((x_plot, std), axis=1)
            update_df = pd.DataFrame(new_trial_stds, columns=["x", "std(y|x)"])
            df = df.append(update_df, ignore_index=True)

    return df


# ------------
# Misc plot methods
# ------------


def kl_grad_shift_plot(ax, model, training_dataset):
    # Unpacking
    x_plot, y_plot, _ = plot_dataset
    x_train, _ = training_dataset

    with torch.set_grad_enabled(True):
        x_train.requires_grad = True
        μ_x, α_x, β_x = model(x_train)
        print(torch.mean(α_x), torch.mean(β_x))
        kl_divergence = model.kl(α_x, β_x, model.prior_α, model.prior_β)
        x_out = model.ood_x(x_train, kl=kl_divergence)
        x_train, x_out = (
            x_train.detach().numpy().flatten(),
            x_out.detach().numpy().flatten(),
        )

    # Reduce clutter by limiting number of points displayed
    N_display = 100

    if x_out is not None and x_out.size > 0:
        ax.scatter(
            np.random.choice(x_out, N_display),
            np.zeros((N_display,)),
            color=colours["primaryRed"],
            alpha=0.5,
            marker="x",
            s=8,
            label=r"$\hat{x}_{n}$",
        )
    ax.scatter(
        np.random.choice(x_train, N_display),
        np.zeros((N_display,)),
        color=colours["navyBlue"],
        alpha=0.5,
        marker="x",
        s=8,
        label=r"$x_{n}$",
    )

    # Plot KL for reference

    # Plot box
    top_kl_plot = 6
    plot_x_range = [data_range_plot[0] - 1, data_range_plot[1] + 1]

    with torch.set_grad_enabled(False):
        # Forward pass
        a, b = fit_prior()
        pp = tcd.Gamma(a, b)
        μ_x, α_x, β_x = model(torch.Tensor(x_plot))
        qp = tcd.Gamma(α_x, β_x)
        kl = tcd.kl_divergence(qp, pp)
        expected_log_lambda = torch.digamma(α_x) - torch.log(β_x)
        expected_lambda = α_x / β_x
        llk = (1 / 2) * (
            expected_log_lambda
            - np.log(2 * np.pi)
            - expected_lambda * ((torch.Tensor(y_plot) - μ_x) ** 2)
        )

    ax.plot(
        x_plot,
        kl,
        "o",
        label=r"KL(q($\lambda$|x)||p($\lambda$))",
        markersize=2,
        markerfacecolor=(*colours_rgb["navyBlue"], 0.6),
        markeredgewidth=1,
        markeredgecolor=(*colours_rgb["navyBlue"], 0.1),
    )
    ax.plot(
        x_plot,
        llk,
        "o",
        label=r"LLK(x,y,$\lambda$)",
        markersize=2,
        markerfacecolor=(*colours_rgb["orange"], 0.6),
        markeredgewidth=1,
        markeredgecolor=(*colours_rgb["orange"], 0.1),
    )

    # Misc
    ax.grid(True)
    ax.set_xlim(plot_x_range)
    ax.set_ylim([-top_kl_plot, top_kl_plot])
    ax.set_xlabel("x")
    ax.legend()

    return ax
