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

        fig, ax = plt.subplots(1, 1)

        x_train, y_train = training_dataset
        ax.plot(
            x_train[:, index],
            y_train,
            "o",
            markersize=3,
            markerfacecolor=(*colours_rgb["navyBlue"], 0.6),
            markeredgewidth=1,
            markeredgecolor=(*colours_rgb["navyBlue"], 0.1),
        )

        # Predictive plot
        # Overextending x_plot
        maxes, _ = torch.max(x_train, dim=0)
        mines, _ = torch.min(x_train, dim=0)
        maxes = torch.where(maxes > 0, 1.25 * maxes, 0.75 * maxes)
        mines = torch.where(mines > 0, 0.75 * mines, 1.25 * mines)
        x_plot = [
            torch.linspace(mines[i], maxes[i], steps=1000)[:, None]
            for i in range(x_train.shape[1])
        ]
        x_plot = torch.cat(x_plot, dim=1)

        for method in methods:
            colour = get_colour_for_method(method)

            best_mean = best_model.predictive_mean(x_plot, method).flatten()
            best_std = best_model.predictive_std(x_plot, method).flatten()

            ax.plot(x_plot[:, index], best_mean, "-", color=colour, alpha=0.55)
            ax.fill_between(
                x_plot[:, index],
                best_mean + 1.96 * best_std,
                best_mean - 1.96 * best_std,
                facecolor=colour,
                alpha=0.3,
            )

        save_folder = f"{experiment_log.save_dir}/{experiment_log.experiment_name}/{experiment_log.name}"
        plt.tight_layout()
        plt.savefig(f"{save_folder}/_i{index}.png")
        plt.show()
