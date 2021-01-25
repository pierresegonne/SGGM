import numpy as np
import pytorch_lightning as pl
import torch

from sggm.definitions import β_OOD
from typing import Any


def split_mean_uncertainty_training(
    experiment: Any,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
) -> (Any, pl.LightningModule, pl.LightningDataModule, pl.Trainer):
    # Heuristic = Purely Eng

    # Set mode to standard ELBO
    # This effectively creates a new var not only a reference
    original_β_OOD = model.β_ood
    model.β_ood = 0

    # Normal fit
    trainer = experiment.trainer

    for cb in trainer.callbacks:
        if isinstance(cb, pl.callbacks.EarlyStopping):
            es = cb

    trainer.fit(model, datamodule)

    # Reset ELBO params
    # Very hacky!
    model.β_ood = original_β_OOD
    es.best_score = torch.tensor(np.Inf)
    es.wait_count = 0
    es.stopped_epoch = 0
    es.patience = es.patience * 2
    trainer.should_stop = False

    # Freeze mean network
    for param in model.μ.parameters():
        param.requires_grad = False
    # Retrain
    model.trainer.current_epoch = 0
    trainer.fit(model, datamodule)
    return experiment, model, datamodule, trainer
