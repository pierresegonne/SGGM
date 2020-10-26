import glob
import numpy as np
import os
import torch
import yaml

from sggm.definitions import regression_experiments
from sggm.regression_model import Regressor


def version_dir(path):
    return os.path.basename(os.path.normpath(path))


class VersionLog:
    def __init__(self, version_dir, version_path, pl_module):
        self.version_dir = version_dir
        self.version_id = int(version_dir.split("_")[-1])
        self.version_path = version_path

        # Load from version
        # model
        checkpoint_name = glob.glob(f"{version_path}/checkpoints/*")[-1]
        self.model = pl_module.load_from_checkpoint(checkpoint_name)
        # performance
        self.results = torch.load(f"{version_path}/results.pkl")
        # datasets
        self.train_dataset = torch.load(f"{version_path}/train_dataset.pkl")
        self.val_dataset = torch.load(f"{version_path}/val_dataset.pkl")
        # hparams
        with open(f"{version_path}/hparams.yaml") as hparams_file:
            self.hparams = yaml.load(hparams_file, Loader=yaml.FullLoader)


class ExperimentLog:
    def __init__(self, experiment_name, name, save_dir="../lightning_logs"):
        self.experiment_name = experiment_name
        self.name = name
        self.save_dir = save_dir

        # Attribute the right PL Module for loading
        if self.experiment_name in regression_experiments:
            pl_module = Regressor

        self.versions = [
            VersionLog(version_dir(version_path), version_path, pl_module)
            for version_path in glob.glob(f"{save_dir}/{experiment_name}/{name}/*/")
        ]
        self.idx_best_version = np.argmin(
            [v.results[0]["test_loss"] for v in self.versions]
        )
        self.test_loss_best_version = self.versions[self.idx_best_version].results[0][
            "test_loss"
        ]
