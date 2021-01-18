import pytorch_lightning as pl

from sggm.definitions import β_OOD
from typing import Any


def split_mean_uncertainty_training(
    experiment: Any,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
) -> (Any, pl.LightningModule, pl.LightningDataModule, pl.Trainer):
    # Heuristic = Purely Eng

    # Set mode to standard ELBO
    original_β_OOD = model.β_ood
    setattr(model, β_OOD, 0)

    # Normal fit
    trainer = experiment.trainer
    trainer.fit(model, datamodule)

    # Reset ELBO params
    setattr(model, β_OOD, original_β_OOD)
    # Freeze mean network
    # print(next(model.μ.parameters())[:5])
    # print(next(model.α.parameters())[:5])
    for param in model.μ.parameters():
        param.requires_grad = False
    # Retrain
    model.trainer.current_epoch = 0
    trainer.fit(model, datamodule)
    # print(next(model.μ.parameters())[:5])
    # print(next(model.α.parameters())[:5])
    return experiment, model, datamodule, trainer
