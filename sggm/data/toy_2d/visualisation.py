import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import seaborn as sns
import torch

from sggm.data.toy_2d.datamodule import Toy2DDataModule


def main():
    dm = Toy2DDataModule(1000, 0)
    dm.setup()

    ds = dm.toy_train.dataset.tensors  # train + val
    x_train, y_train = ds

    slice_idx = (0 < torch.atan(x_train[:, 1] / x_train[:, 0])) & (
        torch.atan(x_train[:, 1] / x_train[:, 0]) < 0.3
    )  # the fact that we look at a slice distords the data a bit
    x_plot = x_train[slice_idx]
    y_plot = y_train[slice_idx]

    x_plot_theoretical = torch.linspace(start=-15, end=15, steps=1000)
    y_plot_theoretical = dm.data_mean(x_plot_theoretical)
    y_plot_std_theoretical = dm.data_std(x_plot_theoretical)

    fig, ax = plt.subplots()
    ax.scatter(x_plot[:, 0], y_plot)
    ax.plot(x_plot_theoretical, y_plot_theoretical, color="black", linewidth=1)
    ax.plot(
        x_plot_theoretical,
        y_plot_theoretical - 1.96 * y_plot_std_theoretical,
        color="black",
        linewidth=1,
        linestyle="dashdot",
    )
    ax.plot(
        x_plot_theoretical,
        y_plot_theoretical + 1.96 * y_plot_std_theoretical,
        color="black",
        linewidth=1,
        linestyle="dashdot",
    )


if __name__ == "__main__":
    main()
    plt.legend()
    plt.show()
