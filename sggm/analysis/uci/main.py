import matplotlib.pyplot as plt
import numpy as np
import random
import torch

from pytorch_lightning import seed_everything
from torch import no_grad

from sggm.analysis.toy.helper import get_colour_for_method
from sggm.data.uci_ccpp.datamodule import UCICCPPDataModule, UCICCPPDataModuleShifted
from sggm.data.uci_concrete import UCIConcreteDataModule, UCIConcreteDataModuleShifted
from sggm.data.uci_superconduct import (
    UCISuperConductDataModule,
    UCISuperConductDataModuleShifted,
)
from sggm.data.uci_wine_red import UCIWineRedDataModule, UCIWineRedDataModuleShifted
from sggm.data.uci_wine_white import (
    UCIWineWhiteDataModule,
    UCIWineWhiteDataModuleShifted,
)
from sggm.data.uci_yacht import UCIYachtDataModule, UCIYachtDataModuleShifted
from sggm.definitions import (
    UCI_CCPP,
    UCI_CONCRETE,
    UCI_SUPERCONDUCT,
    UCI_WINE_RED,
    UCI_WINE_WHITE,
    UCI_YACHT,
)
from sggm.definitions import (
    UCI_CCPP_SHIFTED,
    UCI_CONCRETE_SHIFTED,
    UCI_SUPERCONDUCT_SHIFTED,
    UCI_WINE_RED_SHIFTED,
    UCI_WINE_WHITE_SHIFTED,
    UCI_YACHT_SHIFTED,
)
from sggm.definitions import (
    SEED,
    SHIFTING_PROPORTION_K,
    SHIFTING_PROPORTION_TOTAL,
    GAUSSIAN_NOISE,
)
from sggm.styles_ import colours, colours_rgb, random_rgb_colour


def plot(experiment_log, methods):
    with no_grad():
        best_model = experiment_log.best_version.model

        # Get correct datamodule
        bs = 1000
        experiment_name = experiment_log.experiment_name
        if experiment_name == UCI_CCPP:
            dm = UCICCPPDataModule(bs, 0)
        elif experiment_name == UCI_CONCRETE:
            dm = UCIConcreteDataModule(bs, 0)
        elif experiment_name == UCI_SUPERCONDUCT:
            dm = UCISuperConductDataModule(bs, 0)
        elif experiment_name == UCI_WINE_RED:
            dm = UCIWineRedDataModule(bs, 0)
        elif experiment_name == UCI_WINE_WHITE:
            dm = UCIWineWhiteDataModule(bs, 0)
        elif experiment_name == UCI_YACHT:
            dm = UCIYachtDataModule(bs, 0)
        # Shifted
        elif "_shifted" in experiment_name:

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

            if experiment_name == UCI_CCPP_SHIFTED:
                dm = UCICCPPDataModuleShifted(
                    bs,
                    0,
                    shifting_proportion_total=shifting_proportion_total,
                    shifting_proportion_k=shifting_proportion_k,
                )
            elif experiment_name == UCI_CONCRETE_SHIFTED:
                dm = UCIConcreteDataModuleShifted(
                    bs,
                    0,
                    shifting_proportion_total=shifting_proportion_total,
                    shifting_proportion_k=shifting_proportion_k,
                )
            elif experiment_name == UCI_SUPERCONDUCT_SHIFTED:
                dm = UCISuperConductDataModuleShifted(
                    bs,
                    0,
                    shifting_proportion_total=shifting_proportion_total,
                    shifting_proportion_k=shifting_proportion_k,
                )
            elif experiment_name == UCI_WINE_RED_SHIFTED:
                dm = UCIWineRedDataModuleShifted(
                    bs,
                    0,
                    shifting_proportion_total=shifting_proportion_total,
                    shifting_proportion_k=shifting_proportion_k,
                )
            elif experiment_name == UCI_WINE_WHITE_SHIFTED:
                dm = UCIWineWhiteDataModuleShifted(
                    bs,
                    0,
                    shifting_proportion_total=shifting_proportion_total,
                    shifting_proportion_k=shifting_proportion_k,
                )
            elif experiment_name == UCI_YACHT_SHIFTED:
                dm = UCIYachtDataModuleShifted(
                    bs,
                    0,
                    shifting_proportion_total=shifting_proportion_total,
                    shifting_proportion_k=shifting_proportion_k,
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
        # best_model.ood_x_generation_method = GAUSSIAN_NOISE
        with torch.set_grad_enabled(True):
            x_test.requires_grad = True
            μ_x, α_x, β_x = best_model(x_test)
            kl_divergence = best_model.kl(
                α_x, β_x, best_model.prior_α, best_model.prior_β
            )
            density_lk = best_model.gmm_density(x_test).log_prob(x_test).exp()
            x_out = best_model.ood_x(
                x_test,
                kl=kl_divergence,
                density_lk=density_lk,
            )
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

            mean_train, mean_test = (
                best_model.predictive_mean(x_train, method).flatten(),
                best_model.predictive_mean(x_test, method).flatten(),
            )
            std_train, std_test = (
                best_model.predictive_std(x_train, method).flatten(),
                best_model.predictive_std(x_test, method).flatten(),
            )

            if with_x_out:
                mean_out, std_out = (
                    best_model.predictive_mean(x_out, method).flatten(),
                    best_model.predictive_std(x_out, method).flatten(),
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
                torch.sqrt((y_test.flatten() - mean_test) ** 2),
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

        mean_ax.set_ylabel(r"$y$")
        y_max, _ = torch.max(y_train, dim=0)
        y_min, _ = torch.min(y_train, dim=0)
        y_max = torch.where(y_max > 0, 1.25 * y_max, 0.75 * y_max)
        y_min = torch.where(y_min > 0, 0.75 * y_min, 1.25 * y_min)
        mean_ax.set_ylim((y_min, y_max))
        mean_ax.grid(True)
        mean_ax.legend()

        var_ax.set_ylabel(r"$\sigma$")
        var_ax.set_xlabel(r"$||x||_2$")
        var_ax.grid(True)
        var_ax.legend()

        save_folder = f"{experiment_log.save_dir}/{experiment_log.experiment_name}/{experiment_log.name}"
        plt.tight_layout()
        plt.savefig(f"{save_folder}/_.png")

        plt.show()
