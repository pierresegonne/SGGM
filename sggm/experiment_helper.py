import copy
import numpy as np
import pytorch_lightning as pl
import torch

from sggm.definitions import τ_OOD, SPLIT_TRAINING_MSE_MEAN, SPLIT_TRAINING_STD_VV_MEAN
from torch import nn
from typing import Any


def clean_dict(dic: dict) -> dict:
    clean_dic = {}
    for k, v in dic.items():
        if type(v) in [str, int, bool, float, object, None, list]:
            clean_dic[k] = v
    return clean_dic


def split_mean_uncertainty_training(
    experiment: Any,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
) -> (Any, pl.LightningModule, pl.LightningDataModule, pl.Trainer):
    # Heuristic = Purely Eng

    # Set mode to standard ELBO
    # This effectively creates a new var not only a reference
    # Old
    original_τ_OOD = model.τ_ood
    model.τ_ood = 0
    # New
    # Predict prior uncertainty
    original_α, original_β = model.α, model.β
    original_α_state_dict = copy.deepcopy(model.α.state_dict())
    original_β_state_dict = copy.deepcopy(model.β.state_dict())

    assert model.split_training_mode in [
        SPLIT_TRAINING_MSE_MEAN,
        SPLIT_TRAINING_STD_VV_MEAN,
    ], "Invalid split training mode."

    if model.split_training_mode == SPLIT_TRAINING_MSE_MEAN:
        model.mse_mode = True

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

        model.α = new_α()
        model.β = new_β()

    # Normal fit
    trainer = experiment.trainer

    for cb in trainer.callbacks:
        if isinstance(cb, pl.callbacks.EarlyStopping):
            es = cb

    trainer.fit(model, datamodule)

    # Reset ELBO params
    # Very hacky!
    # Old
    model.τ_ood = original_τ_OOD
    # New
    model.mse_mode = False
    if model.split_training_mode == SPLIT_TRAINING_MSE_MEAN:
        model.α = original_α
        model.β = original_β
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
