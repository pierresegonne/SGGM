import glob
import numpy as np
import os
import torch
import yaml

from sggm.definitions import (
    ENS_REGRESSOR,
    MCD_REGRESSOR,
    VARIATIONAL_REGRESSOR,
    experiments_activation_function,
    experiments_architecture,
    generative_experiments,
    regression_experiments,
    TEST_LOSS,
    VANILLA_VAE,
    VANILLA_VAE_MANIFOLD,
    VV_VAE,
    VV_VAE_MANIFOLD,
    PIG_DL,
    PRIOR_α,
    PRIOR_β,
    INDUCING_CENTROIDS,
)
from sggm.regression_model import VariationalRegressor
from sggm.regression_baselines import ENSRegressor, MCDRegressor
from sggm.vae_model import VanillaVAE, V3AE, V3AEm, VanillaVAEm


def version_dir(path):
    return os.path.basename(os.path.normpath(path))


class VersionLog:
    def __init__(self, experiment_name, version_dir, version_path, pl_module):
        self.version_dir = version_dir
        self.version_id = int(version_dir.split("_")[-1])
        self.version_path = version_path

        # Load from version
        # model
        checkpoint_name = glob.glob(f"{version_path}/checkpoints/*")[-1]
        self.model = pl_module.load_from_checkpoint(
            checkpoint_name,
            activation=experiments_activation_function(experiment_name),
            architecture=experiments_architecture(experiment_name),
            # out_dim=1,
        )
        self.model.freeze()
        # performance
        self.results = torch.load(f"{version_path}/results.pkl")
        # datasets
        self.train_dataset = (
            torch.load(f"{version_path}/train_dataset.pkl")
            if os.path.exists(f"{version_path}/train_dataset.pkl")
            else None
        )
        self.val_dataset = (
            torch.load(f"{version_path}/val_dataset.pkl")
            if os.path.exists(f"{version_path}/val_dataset.pkl")
            else None
        )
        # hparams
        with open(f"{version_path}/hparams.yaml") as hparams_file:
            self.hparams = yaml.load(hparams_file, Loader=yaml.FullLoader)
        # misc
        self.misc = (
            torch.load(f"{version_path}/misc.pkl", map_location=torch.device("cpu"))
            if os.path.exists(f"{version_path}/misc.pkl")
            else None
        )
        if PIG_DL in self.misc:
            self.model.pig_dl = self.misc[PIG_DL]
        if PRIOR_α in self.misc:
            self.model.prior_α = self.misc[PRIOR_α]
        if PRIOR_β in self.misc:
            self.model.prior_β = self.misc[PRIOR_β]
        if INDUCING_CENTROIDS in self.misc:
            self.model.inducing_centroids = self.misc[INDUCING_CENTROIDS]


class ExperimentLog:
    def __init__(
        self, experiment_name, name, model_name=None, save_dir="../lightning_logs"
    ):
        self.experiment_name = experiment_name
        self.name = name
        self.model_name = model_name
        self.save_dir = save_dir

        # Verify the existence of log
        assert os.path.exists(
            f"{save_dir}/{experiment_name}/{name}"
        ), f"experiment {save_dir}/{experiment_name}/{name} is not in logs"

        # Attribute the right PL Module for loading
        if self.experiment_name in regression_experiments:
            pl_module = VariationalRegressor
            if self.model_name == VARIATIONAL_REGRESSOR:
                pl_module = VariationalRegressor
            elif self.model_name == MCD_REGRESSOR:
                pl_module = MCDRegressor
            elif self.model_name == ENS_REGRESSOR:
                pl_module = ENSRegressor
        elif self.experiment_name in generative_experiments:
            if self.model_name == VANILLA_VAE:
                pl_module = VanillaVAE
            elif self.model_name == VANILLA_VAE_MANIFOLD:
                pl_module = VanillaVAEm
            elif self.model_name == VV_VAE:
                pl_module = V3AE
            elif self.model_name == VV_VAE_MANIFOLD:
                pl_module = V3AEm
            else:
                raise NotImplementedError(f"Model {self.model_name} not implemented")
        else:
            raise NotImplementedError(
                f"Experiment {self.experiment_name} is not supported"
            )

        self.versions = [
            VersionLog(
                experiment_name, version_dir(version_path), version_path, pl_module
            )
            for version_path in sorted(
                glob.glob(f"{save_dir}/{experiment_name}/{name}/version_*/"),
                key=lambda v: int(v.split("/")[-2].split("_")[-1]),
            )
        ]
        self.idx_best_version = np.argmin(
            [v.results[0][TEST_LOSS] for v in self.versions]
        )
        self.best_version = self.versions[self.idx_best_version]

    @property
    def experiment_path(self):
        return f"{self.save_dir}/{self.experiment_name}/"
