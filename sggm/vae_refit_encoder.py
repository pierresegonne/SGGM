import argparse
import os
import pytorch_lightning as pl

from copy import deepcopy
from torch import save
import torch
from torch.utils.data import DataLoader

from sggm.analysis.experiment_log import ExperimentLog
from sggm.data.mnist import MNISTDataModule, MNISTDataModuleND
from sggm.data.fashion_mnist import FashionMNISTDataModule, FashionMNISTDataModuleND
from sggm.data.not_mnist import NotMNISTDataModule
from sggm.definitions import (
    experiment_names,
    DIGITS,
)
from sggm.definitions import (
    MNIST,
    MNIST_2D,
    FASHION_MNIST,
    FASHION_MNIST_2D,
    NOT_MNIST,
)
from sggm.vae_model import VanillaVAE, V3AE
from sggm.types_ import Union

"""
Refit your encoder:
Implementation of the workshop paper
"Refit your Encoder when New Data Comes by"
by P.A Mattei and Jes Frellsen
"""

N_EPOCHS_REFIT = 10

#%%


def get_datamodule(
    experiment_name: str, misc: dict, seed=False
) -> pl.LightningDataModule:
    # Get correct datamodule
    bs = 500
    if ("seed" in misc) & seed:
        pl.seed_everything(misc["seed"])
    if experiment_name == MNIST:
        dm = MNISTDataModule(bs, 0)
    elif experiment_name == MNIST_2D:
        if DIGITS in misc:
            dm = MNISTDataModuleND(bs, 0, digits=misc[DIGITS])
        else:
            dm = MNISTDataModuleND(bs, 0)
    elif experiment_name == FASHION_MNIST:
        dm = FashionMNISTDataModule(bs, 0)
    elif experiment_name == FASHION_MNIST_2D:
        if DIGITS in misc:
            dm = FashionMNISTDataModuleND(bs, 0, digits=misc[DIGITS])
        else:
            dm = FashionMNISTDataModuleND(bs, 0)
    elif experiment_name == NOT_MNIST:
        dm = NotMNISTDataModule(bs, 0)
    dm.setup()

    return dm


#%%


# Replace the train dataloader with the data from test.
def train_dataloader(self: pl.LightningDataModule) -> DataLoader:
    return DataLoader(
        self.test_dataset,
        batch_size=self.batch_size,
        num_workers=self.n_workers,
        pin_memory=self.pin_memory,
        shuffle=True,
    )


def refit_encoder(
    model: Union[VanillaVAE, V3AE],
    dm: pl.LightningDataModule,
    logger: pl.loggers.TensorBoardLogger,
):
    # Counterfeit the datamodule
    # https://stackoverflow.com/questions/50599045/python-replacing-a-function-within-a-class-of-a-module
    dm_refit_test = deepcopy(dm)
    dm_refit_test.train_dataloader = train_dataloader.__get__(
        dm_refit_test, pl.LightningDataModule
    )

    # Train
    trainer = pl.Trainer(
        logger=logger,
        automatic_optimization=False,
        max_epochs=N_EPOCHS_REFIT,
    )
    model.train()
    model.freeze_but_encoder()
    for m in model.decoder_α.modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            print("training ? ", m.training)
            break
    if isinstance(model, V3AE):
        model.save_datamodule(dm_refit_test)
        model.set_prior_parameters(
            dm_refit_test, prior_α=model.prior_α, prior_β=model.prior_β
        )
        model.ood_z_generation_method = None

    trainer.fit(model, datamodule=dm_refit_test)

    # Test with refitted encoder
    model.eval()
    res = trainer.test(model, datamodule=dm)

    return res


#%%

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_name", type=str, choices=experiment_names, required=True
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
    )
    parser.add_argument("--run_on", type=str, choices=experiment_names, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="lightning_logs")
    args = parser.parse_args()

    experiment_log = ExperimentLog(
        args.experiment_name,
        args.name,
        model_name=args.model_name,
        save_dir=args.save_dir,
    )
    print(
        f"-- Will refit encoder of best version: {experiment_log.versions[experiment_log.idx_best_version].version_id}"
    )

    # /!\ No check for use on correct experiment/model.
    logger_name = f"{experiment_log.name}_refit_encoder"
    if args.run_on is not None:
        logger_name += "_" + args.run_on
    logger = pl.loggers.TensorBoardLogger(
        save_dir=experiment_log.experiment_path,
        name=logger_name,
    )
    experiment_name = (
        experiment_log.experiment_name if args.run_on is None else args.run_on
    )
    misc = experiment_log.best_version.misc
    dm = get_datamodule(experiment_name, misc)
    res = refit_encoder(experiment_log.best_version.model, dm, logger)

    # Saving
    save(res, f"{logger.log_dir}/results.pkl")
    save(experiment_log.best_version.misc, f"{logger.log_dir}/misc.pkl")
