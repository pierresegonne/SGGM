from typing import Tuple

import torch

from sggm.data.sanity_check import SanityCheckDataModule

# ------------
# Plot data definition
# ------------
data_range_plot = [-15, 25]
data_range_training = [0, 10]


def get_plot_dataset(N: int = 5000) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.linspace(*data_range_plot, steps=N)[:, None]

    return x, SanityCheckDataModule.data_mean(x), SanityCheckDataModule.data_std(x)


plot_dataset = get_plot_dataset()
