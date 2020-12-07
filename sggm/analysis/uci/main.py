import matplotlib.pyplot as plt
import torch

from torch import no_grad

from sggm.analysis.toy.helper import get_colour_for_method
from sggm.data.uci_ccpp.datamodule import UCICCPPDataModule
from sggm.data.uci_concrete import UCIConcreteDataModule
from sggm.data.uci_wine_red import UCIWineRedDataModule
from sggm.data.uci_wine_white import UCIWineWhiteDataModule
from sggm.data.uci_yacht import UCIYachtDataModule
from sggm.definitions import (
    UCI_CONCRETE,
    UCI_CCPP,
    UCI_SUPERCONDUCT,
    UCI_WINE_RED,
    UCI_WINE_WHITE,
    UCI_YACHT,
)
from sggm.styles_ import colours, colours_rgb


def plot(experiment_log, methods, index):
    with no_grad():
        best_model = experiment_log.best_version.model

        # Get correct datamodule
        bs = 10000
        experiment_name = experiment_log.experiment_name
        if experiment_name == UCI_CCPP:
            dm = UCICCPPDataModule(bs, 0)
        elif experiment_name == UCI_CONCRETE:
            dm = UCIConcreteDataModule(bs, 0)
        elif experiment_name == UCI_WINE_RED:
            dm = UCIWineRedDataModule(bs, 0)
        elif experiment_name == UCI_WINE_WHITE:
            dm = UCIWineWhiteDataModule(bs, 0)
        elif experiment_name == UCI_YACHT:
            dm = UCIYachtDataModule(bs, 0)
        dm.setup()

        # Plot training points for reference
        training_dataset = next(iter(dm.train_dataloader()))
        test_dataset = next(iter(dm.test_dataloader()))

        fig, [mean_ax, var_ax] = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        x_train, y_train = training_dataset
        x_test, y_test = test_dataset

        mean_ax.plot(
            x_train[:, index],
            y_train,
            "o",
            markersize=3,
            markerfacecolor=(*colours_rgb["navyBlue"], 0.6),
            markeredgewidth=1,
            markeredgecolor=(*colours_rgb["navyBlue"], 0.1),
            label="Train",
        )
        mean_ax.plot(
            x_test[:, index],
            y_test,
            "o",
            markersize=3,
            markerfacecolor=(*colours_rgb["primaryRed"], 0.6),
            markeredgewidth=1,
            markeredgecolor=(*colours_rgb["primaryRed"], 0.1),
            label="Test",
        )

        # Predictive plot
        # Overextending x_plot
        maxes, _ = torch.max(x_train, dim=0)
        mines, _ = torch.min(x_train, dim=0)
        maxes = torch.where(maxes > 0, 1.25 * maxes, 0.75 * maxes)
        mines = torch.where(mines > 0, 0.75 * mines, 1.25 * mines)
        # This is equivalent of having only one slice of the data.
        x_plot = [
            torch.linspace(mines[i], maxes[i], steps=1000)[:, None]
            for i in range(x_train.shape[1])
        ]
        x_plot = torch.cat(x_plot, dim=1)

        for method in methods:
            colour = get_colour_for_method(method)

            best_mean = best_model.predictive_mean(x_plot, method).flatten()
            best_std = best_model.predictive_std(x_plot, method).flatten()

            mean_ax.plot(
                x_plot[:, index],
                best_mean,
                "-",
                color=colour,
                alpha=0.55,
                label=r"$\mu(x) \pm 1.96\sigma(x)$",
            )
            mean_ax.fill_between(
                x_plot[:, index],
                best_mean + 1.96 * best_std,
                best_mean - 1.96 * best_std,
                facecolor=colour,
                alpha=0.3,
            )

            std_train, std_test = (
                best_model.predictive_std(x_train, method).flatten(),
                best_model.predictive_std(x_test, method).flatten(),
            )
            mean_train, mean_test = (
                best_model.predictive_mean(x_train, method).flatten(),
                best_model.predictive_mean(x_test, method).flatten(),
            )

            var_ax.plot(
                x_train[:, index],
                torch.sqrt((y_train.flatten() - mean_train) ** 2),
                "o",
                markersize=3,
                markerfacecolor=(*colours_rgb["black"], 0.6),
                markeredgewidth=1,
                markeredgecolor=(*colours_rgb["black"], 0.1),
            )
            var_ax.plot(
                x_test[:, index],
                torch.sqrt((y_test.flatten() - mean_test) ** 2),
                "o",
                markersize=3,
                markerfacecolor=(*colours_rgb["black"], 0.6),
                markeredgewidth=1,
                markeredgecolor=(*colours_rgb["black"], 0.1),
                label=r"$\sqrt{(\mu(x)-y)^{2}}$",
            )
            var_ax.plot(
                x_train[:, index],
                std_train,
                "o",
                markersize=3,
                markerfacecolor=(*colours_rgb["orange"], 0.6),
                markeredgewidth=1,
                markeredgecolor=(*colours_rgb["orange"], 0.1),
                label=r"$\sigma(x)$",
            )
            var_ax.plot(
                x_test[:, index],
                std_test,
                "o",
                markersize=3,
                markerfacecolor=(*colours_rgb["orange"], 0.6),
                markeredgewidth=1,
                markeredgecolor=(*colours_rgb["orange"], 0.1),
            )

        mean_ax.set_ylabel("y")
        y_max, _ = torch.max(y_train, dim=0)
        y_min, _ = torch.min(y_train, dim=0)
        y_max = torch.where(y_max > 0, 1.25 * y_max, 0.75 * y_max)
        y_min = torch.where(y_min > 0, 0.75 * y_min, 1.25 * y_min)
        mean_ax.set_ylim((y_min, y_max))
        mean_ax.grid(True)
        mean_ax.legend()

        var_ax.set_ylabel(r"$\sigma$")
        var_ax.set_xlabel(f"x[:{index}]")
        var_ax.grid(True)
        var_ax.legend()

        save_folder = f"{experiment_log.save_dir}/{experiment_log.experiment_name}/{experiment_log.name}"
        plt.tight_layout()
        plt.savefig(f"{save_folder}/_i{index}.png")
        plt.show()
