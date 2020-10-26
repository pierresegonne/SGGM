import matplotlib.pyplot as plt

from torch import no_grad
from sggm.analysis.toy_2d.helper import (
    base_plot_2d,
    best_mean_plot_2d,
    best_std_plot_2d,
    training_plot_3d,
    true_plot_3d,
)


def plot(experiment_log, methods):
    with no_grad():
        best_model = experiment_log.best_version.model
        best_training_dataset = experiment_log.best_version.train_dataset.tensors

        # 3D plotting
        fig_3d = plt.figure()
        ax_3d_training = fig_3d.add_subplot(1, 2, 1, projection="3d")
        ax_3d_true = fig_3d.add_subplot(1, 2, 2, projection="3d")

        # Mean plotting
        fig_mean, [data_ax_mean, std_ax_mean, misc_ax_mean] = plt.subplots(
            3, 1, figsize=(9.5, 9), sharex=True
        )
        fig_mean.suptitle("Mean")

        # Slice plotting
        slice_idx = 0
        fig_slice, [data_ax_slice, std_ax_slice, misc_ax_slice] = plt.subplots(
            3, 1, figsize=(9.5, 9), sharex=True
        )
        fig_slice.suptitle(f"Slice, index={0}")

        # Plot base
        ax_3d_training = training_plot_3d(ax_3d_training, best_training_dataset)
        data_ax_mean, std_ax_mean = base_plot_2d(data_ax_mean, std_ax_mean)
        data_ax_slice, std_ax_slice = base_plot_2d(data_ax_slice, std_ax_slice)

        plt.show()

        for method in methods:
            ax_3d_true = true_plot_3d(ax_3d_true, best_model, method)
            # Rotational mean
            data_ax_mean = best_mean_plot_2d(data_ax_mean, best_model, method)
            # Just one slice
            data_ax_slice = best_mean_plot_2d(
                data_ax_slice, best_model, method, idx=slice_idx
            )

            std_ax_mean = best_std_plot_2d(std_ax_mean, experiment_log, method)
            std_ax_slice = best_std_plot_2d(
                std_ax_slice, experiment_log, method, idx=slice_idx
            )

        plt.legend()
        plt.show()
