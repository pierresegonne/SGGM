import copy
import json
import numpy as np
import os
import pytorch_lightning as pl
import random
import re
import torch
import yaml

from argparse import ArgumentParser, Namespace
from torch.utils.data import DataLoader, random_split

from sggm.callbacks import callbacks
from sggm.definitions import (
    experiment_names,
    generative_experiments,
    generative_models,
    model_names,
    parameters,
    regression_experiments,
    regression_models,
    regressor_parameters,
)
from sggm.definitions import (
    SANITY_CHECK,
    TOY,
    TOY_SHIFTED,
    TOY_2D,
    TOY_2D_SHIFTED,
    UCI_CCPP,
    UCI_CCPP_SHIFTED,
    UCI_CONCRETE,
    UCI_CONCRETE_SHIFTED,
    UCI_SUPERCONDUCT,
    UCI_SUPERCONDUCT_SHIFTED,
    UCI_WINE_RED,
    UCI_WINE_RED_SHIFTED,
    UCI_WINE_WHITE,
    UCI_WINE_WHITE_SHIFTED,
    UCI_YACHT,
    UCI_YACHT_SHIFTED,
    #
    MNIST,
    MNIST_2D,
    FASHION_MNIST,
    FASHION_MNIST_2D,
    NOT_MNIST,
)
from sggm.definitions import (
    VARIATIONAL_REGRESSOR,
    VANILLA_VAE,
    VV_VAE,
    VV_VAE_MANIFOLD,
)
from sggm.definitions import (
    EVAL_LOSS,
    EXPERIMENT_NAME,
    EXPERIMENTS_CONFIG,
    MODEL_NAME,
    PIG_DL,
)
from sggm.definitions import (
    SEED,
    SHIFTING_PROPORTION_K,
    SHIFTING_PROPORTION_TOTAL,
    DIGITS,
    SPLIT_TRAINING,
)
from sggm.definitions import (
    experiments_activation_function,
    experiments_latent_dims,
)
from sggm.data import datamodules
from sggm.experiment_helper import clean_dict, split_mean_uncertainty_training
from sggm.regression_model import Regressor, VariationalRegressor
from sggm.vae_model import BaseVAE, VanillaVAE, V3AE, V3AEm

from sggm.types_ import List


class Experiment:
    def __init__(self, configuration: object):

        # Set dynamically provided configuration attributes
        # Support for config item in 1e-3 notation
        scientific_not_re = re.compile(r"\b-?[1-9](?:\.\d+)?[Ee][-+]?\d+\b")
        for k, v in configuration.items():
            if isinstance(v, str):
                if scientific_not_re.match(v):
                    v = float(v)
            self.__dict__[k] = v

        # Otherwise load default values
        self.add_default_params(parameters)

        # No experiment name
        if getattr(self, EXPERIMENT_NAME, None) is None:
            raise Exception(
                f"Experiment {json.dumps(clean_dict(self.__dict__))} was not supplied any experiment name"
            )

        # And also load default model specific default values
        # Regression
        if self.experiment_name in regression_experiments:
            self.add_default_params(regressor_parameters)

        # Instance attributes
        self._datamodule = None
        self._model = None
        self._callbacks = None
        self._trainer = None

    def add_default_params(self, params):
        for parameter in params.values():
            if parameter.name not in self.__dict__.keys():
                self.__dict__[parameter.name] = parameter.default

    @property
    def model(self):
        # Note: _model is never set,
        # This is because we want to recreate a model everytime it's called
        if self._model is not None:
            return self._model
        else:
            if self.experiment_name in regression_experiments:
                if self.model_name == VARIATIONAL_REGRESSOR:
                    input_dim = self.datamodule.dims
                    return VariationalRegressor(
                        input_dim=input_dim,
                        hidden_dim=self.hidden_dim,
                        activation=experiments_activation_function(
                            self.experiment_name
                        ),
                        learning_rate=self.learning_rate,
                        prior_α=self.prior_alpha,
                        prior_β=self.prior_beta,
                        β_elbo=self.beta_elbo,
                        τ_ood=self.tau_ood,
                        ood_x_generation_method=self.ood_x_generation_method,
                        eps=self.eps,
                        n_mc_samples=self.n_mc_samples,
                        y_mean=self.datamodule.y_mean,
                        y_std=self.datamodule.y_std,
                        split_training_mode=self.split_training_mode,
                        ms_bw_factor=self.ms_bw_factor,
                        ms_kde_bw_factor=self.ms_kde_bw_factor,
                    )
                else:
                    raise NotImplementedError(
                        f"Model {self.model_name} not implemented"
                    )
            elif self.experiment_name in generative_experiments:

                if self.model_name == VANILLA_VAE:
                    return VanillaVAE(
                        input_dims=self.datamodule.dims,
                        activation=experiments_activation_function(
                            self.experiment_name
                        ),
                        latent_dims=experiments_latent_dims(self.experiment_name),
                        learning_rate=self.learning_rate,
                        eps=self.eps,
                        n_mc_samples=self.n_mc_samples,
                    )
                elif self.model_name == VV_VAE:
                    return V3AE(
                        input_dims=self.datamodule.dims,
                        activation=experiments_activation_function(
                            self.experiment_name
                        ),
                        latent_dims=experiments_latent_dims(self.experiment_name),
                        learning_rate=self.learning_rate,
                        prior_α=self.prior_alpha,
                        prior_β=self.prior_beta,
                        τ_ood=self.tau_ood,
                        eps=self.eps,
                        n_mc_samples=self.n_mc_samples,
                        ood_z_generation_method=self.ood_z_generation_method,
                        kde_bandwidth_multiplier=self.kde_bandwidth_multiplier,
                    )
                elif self.model_name == VV_VAE_MANIFOLD:
                    return V3AEm(
                        input_dims=self.datamodule.dims,
                        activation=experiments_activation_function(
                            self.experiment_name
                        ),
                        latent_dims=experiments_latent_dims(self.experiment_name),
                        learning_rate=self.learning_rate,
                        prior_α=self.prior_alpha,
                        prior_β=self.prior_beta,
                        τ_ood=self.tau_ood,
                        eps=self.eps,
                        n_mc_samples=self.n_mc_samples,
                        ood_z_generation_method=self.ood_z_generation_method,
                        kde_bandwidth_multiplier=self.kde_bandwidth_multiplier,
                    )
                else:
                    raise NotImplementedError(
                        f"Model {self.model_name} not implemented"
                    )
            else:
                raise NotImplementedError(
                    f"Experiment {self.experiment_name} is not supported"
                )

    @property
    def datamodule(self):
        if self._datamodule is None:
            self._datamodule = datamodules[self.experiment_name](
                **clean_dict(self.__dict__)
            )
            self._datamodule.setup()
        return self._datamodule

    @property
    def callbacks(self):
        if self._callbacks is None:
            self._callbacks = [
                clbk(**clean_dict(self.__dict__))
                for clbk in callbacks[self.experiment_name]
            ]
        return self._callbacks

    @property
    def trainer(self):
        # Note: _trainer is never set,
        # This is because we want to recreate a model everytime it's called
        if self._trainer is not None:
            return self._trainer
        else:
            # If max_epochs is set to -1, pass on automatic mode
            if self.max_epochs == -1:
                self.max_epochs = self.datamodule.max_epochs

            # Hack to override trainer arguments
            class TrainerArgs:
                def __init__(self, args):
                    self.__dict__ = args

            trainer_args = TrainerArgs(clean_dict(self.__dict__))

            default_callbacks = [
                pl.callbacks.EarlyStopping(
                    EVAL_LOSS, patience=self.early_stopping_patience
                ),
            ]
            # Note that checkpointing is handled by default
            logger = pl.loggers.TensorBoardLogger(
                save_dir=f"lightning_logs/{self.experiment_name}",
                name=self.name,
            )
            automatic_optimization = True
            if self.experiment_name in generative_experiments:
                automatic_optimization = False
            return pl.Trainer.from_argparse_args(
                trainer_args,
                callbacks=self.callbacks + default_callbacks,
                logger=logger,
                profiler=False,
                automatic_optimization=automatic_optimization,
            )


def get_experiments_config(parsed_args: Namespace) -> List[dict]:
    """From the parsed args generate a full experiment config

    Args:
        parsed_args (Namespace): parsed arguments

    Returns:
        List[dict]: List of experiment configs
    """
    experiments_config = [{}]
    if parsed_args.experiments_config:
        with open(parsed_args.experiments_config) as config_file:
            experiments_config = yaml.load(config_file, Loader=yaml.FullLoader)

    parsed_args = vars(parsed_args)
    del parsed_args[f"{EXPERIMENTS_CONFIG}"]

    def _update(a: dict, b: dict) -> dict:
        # The copy is required otherwise all the `a` will point to the same object
        a = copy.deepcopy(a)
        a.update(b)
        return a

    experiments_config = [
        _update(parsed_args, experiment_config)
        for experiment_config in experiments_config
    ]

    return experiments_config


def get_experiment_base_model(experiment_config: dict) -> pl.LightningModule:
    """Extracts experiment name from the experiment config
    and provide base model for that experiment

    Args:
        experiment_config (dict): config of the experiment

    Returns:
        pl.LightningModule: base model for experiment
    """
    assert (EXPERIMENT_NAME in experiment_config.keys()) & (
        experiment_config[EXPERIMENT_NAME] is not None
    ), f"Experiment {experiment_config} was not supplied any experiment name"
    experiment_name = experiment_config[EXPERIMENT_NAME]
    if experiment_name in regression_experiments:
        base_model = Regressor
    if experiment_name in generative_experiments:
        base_model = BaseVAE
    return base_model


def get_model(experiment_config: dict) -> pl.LightningModule:
    assert (MODEL_NAME in experiment_config.keys()) & (
        experiment_config[MODEL_NAME] is not None
    ), f"Experiment {experiment_config} was not supplied any model name"
    model_name = experiment_config[MODEL_NAME]
    if model_name == VARIATIONAL_REGRESSOR:
        model = VariationalRegressor
    elif model_name == VANILLA_VAE:
        model = VanillaVAE
    elif model_name == VV_VAE:
        model = V3AE
    elif model_name == VV_VAE_MANIFOLD:
        model = V3AEm
    else:
        raise NotImplementedError(f"Model {model_name} has no implementation")
    return model


def cli_main():

    # ------------
    # Parse args
    # ------------
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument(f"--{EXPERIMENT_NAME}", choices=experiment_names)
    parser.add_argument(f"--{MODEL_NAME}", choices=model_names)
    parser.add_argument(f"--{EXPERIMENTS_CONFIG}", type=str)
    # Project wide parameters
    for parameter in parameters.values():
        parser.add_argument(
            f"--{parameter.name}",
            default=parameter.default,
            type=parameter.type_,
            choices=parameter.choices,
        )
    args, unknown_args = parser.parse_known_args()

    experiments_config = get_experiments_config(args)

    # ------------
    # Generate experiments
    # ------------
    for experiment_idx, experiment_config in enumerate(experiments_config):

        _experiments_config = copy.deepcopy(experiments_config)
        _experiment_config = copy.deepcopy(experiment_config)
        # Add support for experiment specific arguments
        base_model = get_experiment_base_model(_experiment_config)
        parser = base_model.add_model_specific_args(parser)
        # Reparse with experiment specific arguments
        args, unknown_args = parser.parse_known_args()
        _experiments_config = get_experiments_config(args)

        _experiment_config = _experiments_config[experiment_idx]
        # Add support for model specific arguments
        model = get_model(_experiment_config)
        parser = model.add_model_specific_args(parser)
        # Reparse with model specific arguments
        args, unknown_args = parser.parse_known_args()
        _experiments_config = get_experiments_config(args)

        experiment = Experiment(_experiments_config[experiment_idx])
        print(f"--- Starting Experiment {clean_dict(experiment.__dict__)}")
        for n_t in range(experiment.n_trials):

            if isinstance(experiment.seed, int):
                seed = experiment.seed + n_t
                pl.seed_everything(seed)

            # ------------
            # data
            # ------------
            datamodule = experiment.datamodule
            datamodule.setup()

            shift_investigation = False
            if shift_investigation:
                print(
                    f"{len(datamodule.train_dataset)};{len(datamodule.val_dataset)};{len(datamodule.test_dataset)}"
                )
                continue

            # ------------
            # model
            # ------------
            model = experiment.model
            if isinstance(model, VariationalRegressor):
                model.setup_pig(datamodule)
            if isinstance(model, V3AE):
                model.save_datamodule(datamodule)

            # ------------
            # training
            # ------------
            if getattr(experiment, SPLIT_TRAINING, None):
                (
                    experiment,
                    model,
                    datamodule,
                    trainer,
                ) = split_mean_uncertainty_training(experiment, model, datamodule)
            else:
                trainer = experiment.trainer
                trainer.fit(model, datamodule)

            # ------------
            # testing
            # ------------
            results = trainer.test()

            # ------------
            # saving
            # ------------
            torch.save(results, f"{trainer.logger.log_dir}/results.pkl")
            misc = {}
            if isinstance(experiment.seed, int):
                misc[SEED] = experiment.seed
            if isinstance(experiment.shifting_proportion_k, float):
                misc[SHIFTING_PROPORTION_K] = experiment.shifting_proportion_k
            if isinstance(experiment.shifting_proportion_total, float):
                misc[SHIFTING_PROPORTION_TOTAL] = experiment.shifting_proportion_total
            if getattr(experiment, DIGITS, None):
                misc[DIGITS] = experiment.digits
            if getattr(model, PIG_DL, None):
                misc[PIG_DL] = model.pig_dl
            torch.save(misc, f"{trainer.logger.log_dir}/misc.pkl")


if __name__ == "__main__":
    cli_main()
