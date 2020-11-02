import matplotlib.pyplot as plt

from torch import no_grad
from sggm.analysis.toy.helper import (
    base_plot,
    training_points_plot,
    best_model_plot,
    mean_models_plot,
    # MISC plot functions
    kl_grad_shift_plot,
)


def plot(experiment_log, methods):
    with no_grad():
        best_model = experiment_log.best_version.model
        best_training_dataset = experiment_log.best_version.train_dataset.tensors

        fig, [data_ax, std_ax, misc_ax] = plt.subplots(
            3, 1, figsize=(9.5, 9), sharex=True
        )

        # Start by plotting fixed elements
        data_ax, std_ax = base_plot(data_ax, std_ax)

        # Plot training data for best run
        data_ax = training_points_plot(data_ax, best_training_dataset)

        for method in methods:
            # Plot best run
            data_ax = best_model_plot(data_ax, best_model, method)
            # Plot mean for std
            std_ax = mean_models_plot(std_ax, experiment_log, method)

        # Misc
        misc_ax = kl_grad_shift_plot(
            misc_ax,
            best_model,
            best_training_dataset,
        )

        plt.legend()
        plt.show()
