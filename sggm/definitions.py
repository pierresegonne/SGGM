#!/usr/bin/python
# -*- coding: utf-8 -*-
from typing import Any, List, Union

EXPERIMENT_NAME = "experiment_name"
EXPERIMENTS_CONFIG = "experiments_config"

# All parameters for an experiment must be defined here
BATCH_SIZE = "batch_size"
β_ELBO = "beta_elbo"
β_OOD = "beta_ood"
PRIOR_α = "prior_alpha"
PRIOR_β = "prior_beta"
OOD_X_GENERATION_METHOD = "ood_x_generation_method"
# Options for OOD_X_GENERATION_METHOD
GAUSSIAN_NOISE_AROUND_X = "gaussian_noise_around_x"
OPTIMISED_X_OOD_V_PARAM = "optimised_x_ood_v_param"
OPTIMISED_X_OOD_V_OPTIMISED = "optimised_x_ood_v_optimised"
OPTIMISED_X_OOD_KL_GA = "optimised_x_ood_kl_ga"
OPTIMISED_X_OOD_BRUTE_FORCE = "optimised_x_ood_brute_force"
UNIFORM_X_OOD = "uniform_x_ood"

EPS = "eps"
HIDDEN_DIM = "hidden_dim"
LEARNING_RATE = "learning_rate"
NAME = "name"
N_TRIALS = "n_trials"
N_MC_SAMPLES = "n_mc_samples"
N_WORKERS = "n_workers"  # num workers for data loading set to None will take N_cpus


def none_or_int(value: Any) -> Union[int, None]:
    """Typing for argument being either an int or "None"

    Args:
        value (Any): value whose type will be checked

    Returns:
        Union[int, None]
    """
    if value == "None":
        return None
    return int(value)


def none_or_str(value: Any) -> Union[str, None]:
    """Typing for argument being either a str or "None"

    Args:
        value (Any): value whose type will be checked

    Returns:
        Union[str, None]
    """
    if value == "None":
        return None
    return str(value)


class Param:
    def __init__(
        self,
        name: str,
        default: any,
        type_: type,
        choices: Union[List[Any], None] = None,
    ):
        self.name = name
        self.default = default
        self.type_ = type_
        self.choices = choices


parameters = {
    BATCH_SIZE: Param(BATCH_SIZE, 32, int),
    HIDDEN_DIM: Param(HIDDEN_DIM, 50, int),
    LEARNING_RATE: Param(LEARNING_RATE, 1e-2, float),
    NAME: Param(NAME, "unnamed", str),
    N_TRIALS: Param(N_TRIALS, 1, int),
    N_WORKERS: Param(N_WORKERS, 0, none_or_int),
}

# Parameters specific to the Regressor model
regressor_parameters = {
    β_ELBO: Param(β_ELBO, 1, float),
    β_OOD: Param(β_OOD, 1, float),
    OOD_X_GENERATION_METHOD: Param(
        OOD_X_GENERATION_METHOD,
        None,
        none_or_str,
        choices=[
            GAUSSIAN_NOISE_AROUND_X,
            OPTIMISED_X_OOD_V_PARAM,
            OPTIMISED_X_OOD_V_OPTIMISED,
            OPTIMISED_X_OOD_KL_GA,
            OPTIMISED_X_OOD_BRUTE_FORCE,
            UNIFORM_X_OOD,
        ],
    ),
    EPS: Param(EPS, 1e-10, float),
    N_MC_SAMPLES: Param(N_MC_SAMPLES, 2000, int),
    PRIOR_α: Param(PRIOR_α, 1.05, float),
    PRIOR_β: Param(PRIOR_β, 1.0, float),
}

# -------------
# Experiment Names
# -------------


TOY = "toy"
TOY_2D = "toy_2d"
UCI_SUPERCONDUCT = "uci_superconduct"
UCI_YACHT = "uci_yacht"

experiment_names = [
    TOY,
    TOY_2D,
    UCI_SUPERCONDUCT,
    UCI_YACHT,
]

regression_experiments = [TOY, TOY_2D, UCI_SUPERCONDUCT, UCI_YACHT]

# -------------
# Metric Names
# -------------

TRAIN_LOSS = "train_loss↓"

EVAL_LOSS = "eval_loss↓"

TEST_LOSS = "test_loss↓"
TEST_ELBO = "test_elbo↑"
TEST_MEAN_FIT_MAE = "test_mean_fit_mae↓"
TEST_MEAN_FIT_RMSE = "test_mean_fit_rmse↓"
TEST_VARIANCE_FIT_MAE = "test_variance_fit_mae↓"
TEST_VARIANCE_FIT_RMSE = "test_variance_fit_rmse↓"
TEST_SAMPLE_FIT_MAE = "test_sample_fit_mae↓"
TEST_SAMPLE_FIT_RMSE = "test_sample_fit_rmse↓"
TEST_ELLK = "test_expected_log_likelihood↑"
TEST_KL = "test_kl_divergence↓"
NOISE_ELLK = "noise_expected_log_likelihood↓"
NOISE_KL = "noise_kl_divergence↓"
