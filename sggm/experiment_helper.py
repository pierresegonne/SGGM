import copy
import numpy as np
import pytorch_lightning as pl
import torch

from sggm.definitions import β_OOD
from torch import nn
from typing import Any


def split_mean_uncertainty_training(
    experiment: Any,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
) -> (Any, pl.LightningModule, pl.LightningDataModule, pl.Trainer):
    # Heuristic = Purely Eng

    # Set mode to standard ELBO
    # This effectively creates a new var not only a reference
    # Old
    original_β_OOD = model.β_ood
    model.β_ood = 0
    # New
    # model.mse_mode = True
    # Predict prior uncertainty
    original_α, original_β = model.α, model.β
    original_α_state_dict = copy.deepcopy(model.α.state_dict())
    original_β_state_dict = copy.deepcopy(model.β.state_dict())

    class new_α(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.ones_like(x) * model.prior_α

    class new_β(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.ones_like(x) * model.prior_β

    # model.α = new_α()
    # model.β = new_β()

    # Normal fit
    trainer = experiment.trainer

    for cb in trainer.callbacks:
        if isinstance(cb, pl.callbacks.EarlyStopping):
            es = cb

    trainer.fit(model, datamodule)

    # Reset ELBO params
    # Very hacky!
    # Old
    model.β_ood = original_β_OOD
    # New
    model.mse_mode = False
    model.α.load_state_dict(original_α_state_dict)
    model.β.load_state_dict(original_β_state_dict)
    # ES
    es.best_score = torch.tensor(np.Inf)
    es.wait_count = 0
    es.stopped_epoch = 0
    # es.patience = es.patience * 2 # Should not be necessary if ST works..
    trainer.should_stop = False

    # Freeze mean network
    for param in model.μ.parameters():
        param.requires_grad = False
    # Retrain
    model.trainer.current_epoch = 0
    trainer.fit(model, datamodule)
    return experiment, model, datamodule, trainer
