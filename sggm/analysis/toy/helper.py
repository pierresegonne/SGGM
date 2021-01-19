import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.distributions as tcd

from matplotlib.axes import Axes
from sggm.types_ import Tuple, List
from scipy.stats import norm
from sggm.analysis.experiment_log import ExperimentLog
from sggm.data.toy import ToyDataModule
from sggm.regression_model import (
    MARGINAL,
    POSTERIOR,
    #
    VariationalRegressor,
)
from sggm.styles_ import colours, colours_rgb

# TODO remove once investigation for llk is completed
from sklearn.mixture import GaussianMixture

# ------------
# Plot data definition
# ------------
data_range_plot = [-10, 20]
data_range_training = [0, 10]


def get_plot_dataset(N: int = 5000) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.linspace(*data_range_plot, steps=N)[:, None]

    return x, ToyDataModule.data_mean(x), ToyDataModule.data_std(x)


plot_dataset = get_plot_dataset()


def get_colour_for_method(method: str) -> str:
    if method == MARGINAL:
        return colours["orange"]
    elif method == POSTERIOR:
        return colours["navyBlue"]


# ------------
# Plot methods
# ------------
def base_plot(
    data_ax: Axes,
    std_ax: Axes,
    data: Tuple[torch.Tensor] = plot_dataset,
    range_: List[float] = data_range_training,
    plot_lims={
        "data_ax": [-25, 25],
        "std_ax": [0, 5],
    },
) -> Tuple[Axes]:

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


def training_points_plot(data_ax: Axes, training_dataset: Tuple[torch.Tensor]) -> Axes:
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


def best_model_plot(data_ax: Axes, model: VariationalRegressor, method: str) -> Axes:
    x_plot = plot_dataset[0]
    colour = get_colour_for_method(method)
    best_mean = model.predictive_mean(x_plot, method).flatten()
    best_std = model.predictive_std(x_plot, method).flatten()
    prior_std = model.prior_std(x_plot).flatten()

    # Prior plot
    data_ax.fill_between(
        x_plot.flatten(),
        best_mean + 1.96 * prior_std,
        best_mean - 1.96 * prior_std,
        facecolor=colours["grey"],
        alpha=0.65,
        label=r"$\mu_{%s}(x) \pm 1.96 \,\sigma_{%s}(x)$" % (method, method),
    )

    # Model plot
    data_ax.plot(x_plot, best_mean, "-", color=colour, alpha=0.55)
    data_ax.fill_between(
        x_plot.flatten(),
        best_mean + 1.96 * best_std,
        best_mean - 1.96 * best_std,
        facecolor=colour,
        alpha=0.3,
        label=r"$\mu_{%s}(x) \pm 1.96 \,\sigma_{prior}(x)$" % method,
    )

    data_ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)

    return data_ax


def mean_models_plot(std_ax: Axes, experiment_log: ExperimentLog, method: str) -> Axes:
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
        label=r"$\sigma_{%s}$" % method,
    )

    # Add prior level
    x_plot_prior = torch.Tensor(std_df["x"].values)
    std_prior = experiment_log.best_version.model.prior_std(x_plot_prior)
    std_ax.plot(
        x_plot_prior,
        std_prior,
        color=colours["grey"],
        alpha=0.65,
        label=r"$\sigma_{prior}$",
    )

    std_ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)

    return std_ax


def get_std_trials_df(versions, method: str) -> pd.DataFrame:
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


def kl_grad_shift_plot(
    ax: Axes, model: VariationalRegressor, training_dataset: Tuple[torch.Tensor]
) -> Axes:
    # Unpacking
    x_plot, y_plot, _ = plot_dataset
    x_train, _ = training_dataset

    # Plot X OOD
    with torch.set_grad_enabled(True):
        x_train.requires_grad = True
        μ_x, α_x, β_x = model(x_train)
        kl_divergence = model.kl(α_x, β_x, model.prior_α, model.prior_β)
        density_lk = model.gmm_density(x_train).log_prob(x_train).exp()
        x_out = model.ood_x(
            x_train,
            kl=kl_divergence,
            density_lk=density_lk,
        )
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
    top_kl_plot = 3.5
    plot_x_range = [data_range_plot[0] - 1, data_range_plot[1] + 1]

    with torch.set_grad_enabled(False):
        # Forward pass
        μ_x, α_x, β_x = model(torch.Tensor(x_plot))
        kl = model.kl(α_x, β_x, model.prior_α, model.prior_β)
        ellk = model.ellk(μ_x, α_x, β_x, torch.Tensor(y_plot))
        mllk = tcd.StudentT(2 * α_x, μ_x, torch.sqrt(β_x / α_x)).log_prob(y_plot)

        # TODO likelihood remove once study over
        gm = GaussianMixture(n_components=5).fit(x_train.reshape(-1, 1))
        llk = np.exp(gm.score_samples(x_plot.reshape(-1, 1))).reshape(-1, 1)
        kl_llk = kl - llk

    # KL
    ax.plot(
        x_plot,
        kl,
        "o",
        label=r"KL(q($\lambda\mid$x)$\Vert$p($\lambda$))",
        markersize=2,
        markerfacecolor=(*colours_rgb["navyBlue"], 0.6),
        markeredgewidth=1,
        markeredgecolor=(*colours_rgb["navyBlue"], 0.1),
    )
    # # ELLK
    ax.plot(
        x_plot,
        ellk,
        "o",
        label=r"ELLK(x,y,$\lambda$)",
        markersize=2,
        markerfacecolor=(*colours_rgb["orange"], 0.6),
        markeredgewidth=1,
        markeredgecolor=(*colours_rgb["orange"], 0.1),
    )
    # # MLLK
    # ax.plot(
    #     x_plot,
    #     mllk,
    #     "o",
    #     label=r"MLLK(x,y)",
    #     markersize=2,
    #     markerfacecolor=(*colours_rgb["red"], 0.6),
    #     markeredgewidth=1,
    #     markeredgecolor=(*colours_rgb["red"], 0.1),
    # )
    # # KL penalised by density llk
    # ax.plot(
    #     x_plot,
    #     kl_llk,
    #     "o",
    #     label=r"KL(x)-LLK(x)",
    #     markersize=2,
    #     markerfacecolor=(*colours_rgb["brightGreen"], 0.6),
    #     markeredgewidth=1,
    #     markeredgecolor=(*colours_rgb["brightGreen"], 0.1),
    # )
    # # Density likelihood
    # ax.plot(
    #     x_plot,
    #     llk,
    #     "o",
    #     label=r"LLK(x)",
    #     markersize=2,
    #     markerfacecolor=(*colours_rgb["purple"], 0.6),
    #     markeredgewidth=1,
    #     markeredgecolor=(*colours_rgb["purple"], 0.1),
    # )

    # Gamma parameters
    # ax.plot(
    #     x_plot,
    #     α_x,
    #     "o",
    #     label=r"$\alpha$(x)",
    #     markersize=2,
    #     markerfacecolor=(*colours_rgb["pink"], 0.6),
    #     markeredgewidth=1,
    #     markeredgecolor=(*colours_rgb["pink"], 0.1),
    # )
    # ax.plot(
    #     x_plot,
    #     β_x,
    #     "o",
    #     label=r"$\beta$(x)",
    #     markersize=2,
    #     markerfacecolor=(*colours_rgb["purple"], 0.6),
    #     markeredgewidth=1,
    #     markeredgecolor=(*colours_rgb["purple"], 0.1),
    # )

    # Gamma split aleatoric epistemic
    # ax.plot(
    #     x_plot,
    #     β_x / α_x,
    #     "o",
    #     label=r"$\sigma_{aleatoric}(x) = \frac{\beta(x)}{\alpha(x)}$",
    #     markersize=2,
    #     markerfacecolor=(*colours_rgb["pink"], 0.6),
    #     markeredgewidth=1,
    #     markeredgecolor=(*colours_rgb["pink"], 0.1),
    # )
    # ax.plot(
    #     x_plot,
    #     α_x / (α_x - 1),
    #     "o",
    #     label=r"$\sigma_{epistemic}(x) = \frac{\alpha(x)}{\alpha(x) - 1}$",
    #     markersize=2,
    #     markerfacecolor=(*colours_rgb["purple"], 0.6),
    #     markeredgewidth=1,
    #     markeredgecolor=(*colours_rgb["purple"], 0.1),
    # )
    # ax.plot(
    #     x_plot,
    #     α_x / β_x,
    #     "o",
    #     label=r"$\frac{\alpha(x)}{\beta(x)}$",
    #     markersize=2,
    #     markerfacecolor=(*colours_rgb["primaryRed"], 0.6),
    #     markeredgewidth=1,
    #     markeredgecolor=(*colours_rgb["primaryRed"], 0.1),
    # )

    # Misc
    ax.grid(True)
    ax.set_xlim(plot_x_range)
    ax.set_ylim([-top_kl_plot, top_kl_plot])
    ax.set_xlabel("x")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)

    return ax
