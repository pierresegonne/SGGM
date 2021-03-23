import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.distributions as D

from typing import Tuple
from mpl_toolkits.mplot3d import Axes3D
from sggm.analysis.toy.helper import base_plot, get_colour_for_method
from sggm.data.toy_2d import Toy2DDataModule
from sggm.regression_model import (
    check_available_methods,
    MARGINAL,
    POSTERIOR,
)
from sggm.styles_ import colours, colours_rgb


pi = np.pi

# ------------
# Plot data definition
# ------------
data_range_training = [0, 14]
data_range_plot = [0, 18]


def get_plot_dataset(N: int = 360) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r = torch.linspace(*data_range_plot, steps=N)
    theta = torch.linspace(start=0, end=2 * pi, steps=N)
    r, theta = torch.meshgrid(r.flatten(), theta.flatten())
    r, theta = torch.transpose(r, 0, 1), torch.transpose(theta, 0, 1)
    x, y = r * torch.cos(theta), r * torch.sin(theta)

    x_plot = torch.cat([x.flatten()[:, None], y.flatten()[:, None]], 1)

    r_plot = torch.norm(x_plot, dim=1)

    return x_plot, Toy2DDataModule.data_mean(r_plot), Toy2DDataModule.data_std(r_plot)


plot_dataset = get_plot_dataset()


# ------------
# 3D manipulations
# ------------
def column_to_mesh(c):
    N = int(torch.sqrt(torch.Tensor([c.shape[0]])))
    return torch.reshape(c, (N, N))


def column_to_slice(c, idx=0):
    N = int(torch.sqrt(torch.Tensor([c.shape[0]])))
    return torch.reshape(c, (N, N))[idx, :]


# ------------
# Plot methods
# ------------
def base_plot_true_3d(ax):
    # Unpacking
    x_plot, y_plot, _ = plot_dataset

    # 2D mesh for 3D plot
    X_plot_mesh = column_to_mesh(x_plot[:, 0])
    Y_plot_mesh = column_to_mesh(x_plot[:, 1])
    Z_plot_mesh = column_to_mesh(y_plot)

    Z_plot_mesh_in_training = torch.where(
        torch.sqrt(X_plot_mesh ** 2 + Y_plot_mesh ** 2) <= data_range_training[1],
        Z_plot_mesh,
        torch.Tensor([float("NaN")]),
    )
    Z_plot_mesh_outside_training = torch.where(
        torch.sqrt(X_plot_mesh ** 2 + Y_plot_mesh ** 2) > data_range_training[1],
        Z_plot_mesh,
        torch.Tensor([float("NaN")]),
    )

    # Plot true mean
    sampling_count = 20
    ax.plot_wireframe(
        X_plot_mesh,
        Y_plot_mesh,
        Z_plot_mesh_in_training,
        ccount=sampling_count,
        rcount=sampling_count,
        color="black",
        linewidth=1,
        alpha=0.65,
    )
    ax.plot_wireframe(
        X_plot_mesh,
        Y_plot_mesh,
        Z_plot_mesh_outside_training,
        ccount=sampling_count,
        rcount=sampling_count,
        color="black",
        linewidth=1,
        alpha=0.65,
        linestyle="dashed",
    )
    ax.set_zlim([-50, 50])

    return ax


def training_plot_3d(ax, training_dataset):
    ax = base_plot_true_3d(ax)

    # Unpacking
    x_train, y_train = training_dataset

    # Plot training points
    ax.plot(
        x_train[:, 0],
        x_train[:, 1],
        y_train[:, 0],
        "o",
        markersize=3,
        markerfacecolor=(*colours_rgb["navyBlue"], 0.6),
        markeredgewidth=1,
        markeredgecolor=(*colours_rgb["navyBlue"], 0.1),
    )

    return ax


def base_plot_2d(data_ax, std_ax):
    x_plot, y_plot, y_std_plot = plot_dataset

    # For y = 0
    x_plot_slice = column_to_slice(x_plot[:, 0])
    y_plot_slice = column_to_slice(y_plot)
    y_std_plot_slice = column_to_slice(y_std_plot)

    data_ax, std_ax = base_plot(
        data_ax,
        std_ax,
        data=(x_plot_slice, y_plot_slice, y_std_plot_slice),
        range_=data_range_training,
        plot_lims={
            "data_ax": [-30, 30],
            "std_ax": [0, 7],
        },
    )
    return data_ax, std_ax


def true_plot_3d(ax, best_model, method):
    ax = base_plot_true_3d(ax)

    # Unpacking
    x_plot, y_plot, _ = plot_dataset

    # Getting best mean
    best_mean = best_model.predictive_mean(x_plot, method)

    # 2D mesh for 3D plot
    X_plot_mesh = column_to_mesh(x_plot[:, 0]).numpy()
    Y_plot_mesh = column_to_mesh(x_plot[:, 1]).numpy()
    Z_plot_mesh = column_to_mesh(best_mean).numpy()

    ax.plot_surface(X_plot_mesh, Y_plot_mesh, Z_plot_mesh, cmap="bone")

    return ax


def best_mean_plot_2d(ax, best_model, method, idx=None):
    assert (idx is None) or isinstance(idx, int)
    # Unpacking
    x_plot, _, _ = plot_dataset
    colour = get_colour_for_method(method)

    # Getting best mean & std
    best_mean = best_model.predictive_mean(x_plot, method)
    best_std = best_model.predictive_std(x_plot, method)

    # Slicing
    x_plot_slice = column_to_slice(x_plot[:, 0])
    if idx is not None:
        best_mean_slice = column_to_slice(best_mean, idx=idx)
        best_std_slice = column_to_slice(best_std, idx=idx)
    else:
        best_mean_slice = torch.mean(column_to_mesh(best_mean), dim=0)
        best_std_slice = torch.mean(column_to_mesh(best_std), dim=0)

    ax.plot(x_plot_slice, best_mean_slice, "-", color=colour, alpha=0.55)
    ax.fill_between(
        x_plot_slice,
        best_mean_slice + 1.96 * best_std_slice,
        best_mean_slice - 1.96 * best_std_slice,
        facecolor=colour,
        alpha=0.3,
    )

    return ax


def best_std_plot_2d(ax, experiment_log, method, idx=None):
    assert (idx is None) or isinstance(idx, int)
    colour = get_colour_for_method(method)
    # Get df for all trials for the std
    std_df = get_std_trials_df(experiment_log.versions, method=method, idx=idx)
    ax = sns.lineplot(
        x="x", y="std(y|x)", ci="sd", data=std_df, ax=ax, color=colour, alpha=0.55
    )

    return ax


def get_std_trials_df(versions, method, idx=None):
    assert method in ["posterior", "marginal"]

    with torch.no_grad():
        x_plot, _, _ = plot_dataset

        df = pd.DataFrame(columns=["x", "std(y|x)"])

        for v in versions:
            std = v.model.predictive_std(x_plot, method)

            if idx is not None:
                std = column_to_slice(std, idx=idx).numpy().reshape((-1, 1))
            else:
                std = column_to_mesh(std).mean(axis=0).numpy().reshape((-1, 1))

            x_plot_slice = column_to_slice(x_plot[:, 0]).numpy().reshape((-1, 1))

            new_trial_stds = np.concatenate((x_plot_slice, std), axis=1).astype(
                np.float64
            )
            update_df = pd.DataFrame(new_trial_stds, columns=["x", "std(y|x)"])
            df = df.append(update_df, ignore_index=True)

    return df
