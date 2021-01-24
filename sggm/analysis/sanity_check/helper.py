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
from sggm.data.sanity_check import SanityCheckDataModule
from sggm.regression_model import (
    MARGINAL,
    POSTERIOR,
    #
    VariationalRegressor,
)
from sggm.styles_ import colours, colours_rgb

# ------------
# Plot data definition
# ------------
data_range_plot = [-15, 25]
data_range_training = [0, 10]


def get_plot_dataset(N: int = 5000) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.linspace(*data_range_plot, steps=N)[:, None]

    return x, SanityCheckDataModule.data_mean(x), SanityCheckDataModule.data_std(x)


plot_dataset = get_plot_dataset()


# ------------
# Plot methods
# ------------
