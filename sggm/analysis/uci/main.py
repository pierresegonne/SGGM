import matplotlib.pyplot as plt
import torch

from pytorch_lightning import seed_everything
from torch import no_grad

from sggm.analysis.toy.helper import get_colour_for_method
from sggm.data import datamodules
from sggm.definitions import (
    SEED,
    SHIFTING_PROPORTION_K,
    SHIFTING_PROPORTION_TOTAL,
    GAUSSIAN_NOISE,
)
from sggm.styles_ import colours, colours_rgb


def plot(experiment_log, methods, show_plot=True):
    # %
    y_idx = 0
    # %

    with no_grad():
        best_model = experiment_log.best_version.model

        # Get correct datamodule
        bs = 1000
        experiment_name = experiment_log.experiment_name

        # Shifted
        if "_shifted" in experiment_name:

            assert (
                experiment_log.best_version.misc is not None
            ), f"Missing misc dictionary for {experiment_log.best_version.version_path}"
            seed = experiment_log.best_version.misc[SEED]
            shifting_proportion_k = experiment_log.best_version.misc[
                SHIFTING_PROPORTION_K
            ]
            shifting_proportion_total = experiment_log.best_version.misc[
                SHIFTING_PROPORTION_TOTAL
            ]
            seed_everything(seed)

            dm = datamodules[experiment_name](
                bs,
                0,
                shifting_proportion_total=shifting_proportion_total,
                shifting_proportion_k=shifting_proportion_k,
            )

        else:
            dm = datamodules[experiment_name](
                bs,
                0,
            )
        dm.setup()

        # Plot training points for reference
        training_dataset = next(iter(dm.train_dataloader()))
        test_dataset = next(iter(dm.test_dataloader()))

        x_train, y_train = training_dataset
        x_test, y_test = test_dataset
        norm_x_test = torch.norm(x_test, dim=1)[:, None]

        # Test ood
        # Fallback on GN generation if no x ood
        best_model.τ_ood = 0.5
        best_model.ood_x_generation_method = GAUSSIAN_NOISE
        x_out = best_model.ood_x(x_test)
        x_test = x_test.detach()
        x_out = x_out.detach()
        with_x_out = True if (x_out is not None and torch.numel(x_out) > 0) else False
        norm_x_out = torch.norm(x_out, dim=1)[:, None]

        # ======================
        # Plot

        # Fit plot
        fig, [mean_ax, var_ax] = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        mean_ax.plot(
            norm_x_test,
            y_test,
            "o",
            markersize=3,
            markerfacecolor=(*colours_rgb["navyBlue"], 0.6),
            markeredgewidth=1,
            markeredgecolor=(*colours_rgb["navyBlue"], 0.1),
            label=r"$y|x$",
        )

        for method in methods:
            colour = get_colour_for_method(method)

            # With multivariate predictions, creates the need to select y dimension
            mean_train, mean_test = (
                best_model.predictive_mean(x_train, method)[:, y_idx].flatten(),
                best_model.predictive_mean(x_test, method)[:, y_idx].flatten(),
            )
            std_train, std_test = (
                best_model.predictive_std(x_train, method)[:, y_idx].flatten(),
                best_model.predictive_std(x_test, method)[:, y_idx].flatten(),
            )

            if with_x_out:
                mean_out, std_out = (
                    best_model.predictive_mean(x_out, method)[:, y_idx].flatten(),
                    best_model.predictive_std(x_out, method)[:, y_idx].flatten(),
                )

            mean_ax.errorbar(
                norm_x_test,
                mean_test,
                yerr=1.96 * std_test,
                fmt="o",
                color=colour,
                markersize=3,
                elinewidth=1,
                alpha=0.55,
                label=r"$\mu(x)\pm\,1.96\,\sigma(x)$",
            )
            if with_x_out:
                mean_ax.errorbar(
                    norm_x_out,
                    mean_out,
                    yerr=1.96 * std_out,
                    fmt="o",
                    color=colours["primaryRed"],
                    markersize=3,
                    elinewidth=1,
                    alpha=0.55,
                    label=r"$\mu(\hat{x})\pm\,1.96\,\sigma(\hat{x})$",
                )

            var_ax.plot(
                norm_x_test,
                torch.sqrt((y_test[:, y_idx].flatten() - mean_test) ** 2),
                "o",
                markersize=3,
                markerfacecolor=(*colours_rgb["navyBlue"], 0.6),
                markeredgewidth=1,
                markeredgecolor=(*colours_rgb["navyBlue"], 0.1),
                label=r"$\sqrt{(\mu(x)-y)^{2}}$",
            )
            var_ax.plot(
                norm_x_test,
                std_test,
                "o",
                markersize=3,
                markerfacecolor=(*colours_rgb["orange"], 0.6),
                markeredgewidth=1,
                markeredgecolor=(*colours_rgb["orange"], 0.1),
                label=r"$\sigma(x)$",
            )
            var_ax.plot(
                norm_x_test,
                best_model.prior_std(x_test),
                "o",
                markersize=3,
                markerfacecolor=(*colours_rgb["black"], 0.4),
                markeredgewidth=1,
                markeredgecolor=(*colours_rgb["black"], 0.1),
                label=r"$\sigma_{prior}(x)$",
            )
            if with_x_out:
                var_ax.plot(
                    norm_x_out,
                    std_out,
                    "o",
                    markersize=3,
                    markerfacecolor=(*colours_rgb["primaryRed"], 0.6),
                    markeredgewidth=1,
                    markeredgecolor=(*colours_rgb["primaryRed"], 0.1),
                    label=r"$\sigma(\hat{x})$",
                )
                var_ax.plot(
                    norm_x_out,
                    best_model.prior_std(x_out),
                    "o",
                    markersize=3,
                    markerfacecolor=(*colours_rgb["black"], 0.4),
                    markeredgewidth=1,
                    markeredgecolor=(*colours_rgb["black"], 0.1),
                    label=r"$\sigma_{prior}(\hat{x})$",
                )

            print(f"Average test uncertainty= {std_test.mean():.3f}")
            if with_x_out:
                print(f"Average OOD uncertainty= {std_out.mean():.3f}")

        mean_ax.set_ylabel(r"$y$")
        y_max, _ = torch.max(y_train, dim=0)
        y_min, _ = torch.min(y_train, dim=0)
        y_max = torch.where(y_max > 0, 1.25 * y_max, 0.75 * y_max)
        y_min = torch.where(y_min > 0, 0.75 * y_min, 1.25 * y_min)
        mean_ax.set_ylim((y_min[y_idx], y_max[y_idx]))
        mean_ax.grid(True)
        mean_ax.legend()

        var_ax.set_ylabel(r"$\sigma$")
        var_ax.set_xlabel(r"$||x||_2$")
        var_ax.grid(True)
        var_ax.legend()

        save_folder = f"{experiment_log.save_dir}/{experiment_log.experiment_name}/{experiment_log.name}"
        plt.tight_layout()
        plt.savefig(f"{save_folder}/_.png")

        if show_plot:
            plt.show()
        plt.close()
