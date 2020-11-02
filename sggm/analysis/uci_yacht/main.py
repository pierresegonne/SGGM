import matplotlib.pyplot as plt
import torch

from torch import no_grad
from sggm.styles_ import colours, colours_rgb


data_range_plot = [-2, 2]


def get_plot_dataset(N: int = 1000) -> torch.Tensor:
    x_l = [torch.linspace(*data_range_plot, steps=N)[:, None] for _ in range(6)]
    x = torch.cat(x_l, dim=1)
    print(x.shape)
    return x


x_plot = get_plot_dataset()

method = "marginal"


def plot(experiment_log, methods):
    with no_grad():
        best_model = experiment_log.best_version.model
        best_training_dataset = experiment_log.best_version.train_dataset.tensors

        fig, ax = plt.subplots(1, 1)

        x_train, y_train = best_training_dataset
        ax.plot(
            x_train[:, -1],
            y_train,
            "o",
            markersize=3,
            markerfacecolor=(*colours_rgb["navyBlue"], 0.6),
            markeredgewidth=1,
            markeredgecolor=(*colours_rgb["navyBlue"], 0.1),
        )

        best_mean = best_model.predictive_mean(x_plot, method).flatten()
        best_std = best_model.predictive_std(x_plot, method).flatten()

        ax.plot(x_plot, best_mean, "-", color=colours["orange"], alpha=0.55)
        ax.fill_between(
            x_plot[:, -1].flatten(),
            best_mean + 1.96 * best_std,
            best_mean - 1.96 * best_std,
            facecolor=colours["orange"],
            alpha=0.3,
        )

        plt.legend()
        plt.show()
