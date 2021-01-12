from argparse import ArgumentParser
from typing import Any, List, Union

EXPERIMENT_NAME = "experiment_name"
EXPERIMENTS_CONFIG = "experiments_config"
MODEL_NAME = "model_name"

# -------------
# Parameters
# -------------

# All parameters for an experiment must be defined here
SEED = "seed"
BATCH_SIZE = "batch_size"
β_ELBO = "beta_elbo"
β_OOD = "beta_ood"
PRIOR_α = "prior_alpha"
PRIOR_β = "prior_beta"
OOD_X_GENERATION_METHOD = "ood_x_generation_method"
# Options for OOD_X_GENERATION_METHOD
GAUSSIAN_NOISE = "gaussian_noise"
V_PARAM = "v_param"
ADVERSARIAL = "adversarial"
BRUTE_FORCE = "brute_force"
UNIFORM = "uniform"

OOD_X_GENERATION_AVAILABLE_METHODS = [
    GAUSSIAN_NOISE,
    V_PARAM,
    ADVERSARIAL,
    BRUTE_FORCE,
    UNIFORM,
]

EPS = "eps"
EARLY_STOPPING_PATIENCE = "early_stopping_patience"
HIDDEN_DIM = "hidden_dim"
LEARNING_RATE = "learning_rate"
NAME = "name"
N_TRIALS = "n_trials"
N_MC_SAMPLES = "n_mc_samples"
N_WORKERS = "n_workers"  # num workers for data loading set to None will take N_cpus

SHIFTING_PROPORTION_TOTAL = "shifting_proportion_total"
SHIFTING_PROPORTION_K = "shifting_proportion_k"

ENCODER_TYPE = "encoder_type"
# Options for ENCODER_TYPE
ENCODER_FULLY_CONNECTED = "encoder_fully_connected"
ENCODER_CONVOLUTIONAL = "encoder_convolutional"
ENCODER_AVAILABLE_TYPES = [ENCODER_FULLY_CONNECTED, ENCODER_CONVOLUTIONAL]


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
    NAME: Param(NAME, "unnamed", str),
    N_TRIALS: Param(N_TRIALS, 1, int),
    N_WORKERS: Param(N_WORKERS, 0, none_or_int),
    EARLY_STOPPING_PATIENCE: Param(EARLY_STOPPING_PATIENCE, 50, int),
    SEED: Param(SEED, None, none_or_int),
    #
    SHIFTING_PROPORTION_TOTAL: Param(SHIFTING_PROPORTION_TOTAL, 1e-1, float),
    SHIFTING_PROPORTION_K: Param(SHIFTING_PROPORTION_K, 1e-2, float),
}

# -------------
# Models
# -------------
VARIATIONAL_REGRESSOR = "variational_regressor"
regressor_parameters = {
    MODEL_NAME: Param(
        MODEL_NAME, VARIATIONAL_REGRESSOR, str, choices=[VARIATIONAL_REGRESSOR]
    ),
}
variational_regressor_parameters = {
    β_ELBO: Param(β_ELBO, 1, float),
    β_OOD: Param(β_OOD, 1, float),
    LEARNING_RATE: Param(LEARNING_RATE, 1e-2, float),
    EPS: Param(EPS, 1e-10, float),
    HIDDEN_DIM: Param(HIDDEN_DIM, 50, int),
    N_MC_SAMPLES: Param(N_MC_SAMPLES, 20, int),
    OOD_X_GENERATION_METHOD: Param(
        OOD_X_GENERATION_METHOD,
        None,
        none_or_str,
        choices=OOD_X_GENERATION_AVAILABLE_METHODS,
    ),
    PRIOR_α: Param(PRIOR_α, 1.05, float),
    PRIOR_β: Param(PRIOR_β, 1.0, float),
}
regression_models = [VARIATIONAL_REGRESSOR]

VANILLA_VAE = "vanilla_vae"
VV_VAE = "v3ae"
vae_parameters = {
    ENCODER_TYPE: Param(
        ENCODER_TYPE, ENCODER_FULLY_CONNECTED, str, choices=ENCODER_AVAILABLE_TYPES
    ),
}
vanilla_vae_parameters = {}
v3ae_parameters = {}
generative_models = [VANILLA_VAE, VV_VAE]


model_names = regression_models + generative_models


def model_specific_args(params, parent_parser):
    parser = ArgumentParser(
        parents=[parent_parser], add_help=False, conflict_handler="resolve"
    )
    for parameter in params.values():
        parser.add_argument(
            f"--{parameter.name}", default=parameter.default, type=parameter.type_
        )
    return parser


# -------------
# Experiment Names
# -------------

SHIFTED = "_shifted"
# Regression
SANITY_CHECK = "sanity_check"
TOY = "toy"
TOY_SHIFTED = TOY + SHIFTED
TOY_2D = "toy_2d"
TOY_2D_SHIFTED = TOY_2D + SHIFTED
UCI_CCPP = "uci_ccpp"
UCI_CCPP_SHIFTED = UCI_CCPP + SHIFTED
UCI_CONCRETE = "uci_concrete"
UCI_CONCRETE_SHIFTED = UCI_CONCRETE + SHIFTED
UCI_SUPERCONDUCT = "uci_superconduct"
UCI_SUPERCONDUCT_SHIFTED = UCI_SUPERCONDUCT + SHIFTED
UCI_WINE_RED = "uci_wine_red"
UCI_WINE_RED_SHIFTED = UCI_WINE_RED + SHIFTED
UCI_WINE_WHITE = "uci_wine_white"
UCI_WINE_WHITE_SHIFTED = UCI_WINE_WHITE + SHIFTED
UCI_YACHT = "uci_yacht"
UCI_YACHT_SHIFTED = UCI_YACHT + SHIFTED
# Generative
MNIST = "mnist"
FASHION_MNIST = "fashion_mnist"
NOT_MNIST = "not_mnist"


experiment_names = [
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
    MNIST,
    FASHION_MNIST,
    NOT_MNIST,
]
regression_experiments = [
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
]
generative_experiments = [MNIST, FASHION_MNIST, NOT_MNIST]


# -------------
# Max batch iterations
# -------------
SANITY_CHECK_MAX_BATCH_ITERATIONS = 6e3
TOY_MAX_BATCH_ITERATIONS = 6e3
TOY_2D_MAX_BATCH_ITERATIONS = 1e4
UCI_LARGE_MAX_BATCH_ITERATIONS = 1e5
UCI_SMALL_MAX_BATCH_ITERATIONS = 2e4


# -------------
# Activation functions
# -------------
F_ELU = "elu"
F_LEAKY_RELU = "leaky_relu"
F_RELU = "relu"
F_SIGMOID = "sigmoid"

ACTIVATION_FUNCTIONS = [
    F_ELU,
    F_LEAKY_RELU,
    F_RELU,
    F_SIGMOID,
]


# -------------
# Metric Names
# -------------
UP_METRIC_INDICATOR = "↑"
DOWN_METRIC_INDICATOR = "↓"

TRAIN_LOSS = f"train_loss{DOWN_METRIC_INDICATOR}"

EVAL_LOSS = f"eval_loss{DOWN_METRIC_INDICATOR}"

TEST_LOSS = f"test_loss{DOWN_METRIC_INDICATOR}"
TEST_ELBO = f"test_elbo{UP_METRIC_INDICATOR}"
TEST_MLLK = f"test_marginal_log_likelihood{UP_METRIC_INDICATOR}"
TEST_MEAN_FIT_MAE = f"test_mean_fit_mae{DOWN_METRIC_INDICATOR}"
TEST_MEAN_FIT_RMSE = f"test_mean_fit_rmse{DOWN_METRIC_INDICATOR}"
TEST_VARIANCE_FIT_MAE = f"test_variance_fit_mae{DOWN_METRIC_INDICATOR}"
TEST_VARIANCE_FIT_RMSE = f"test_variance_fit_rmse{DOWN_METRIC_INDICATOR}"
TEST_SAMPLE_FIT_MAE = f"test_sample_fit_mae{DOWN_METRIC_INDICATOR}"
TEST_SAMPLE_FIT_RMSE = f"test_sample_fit_rmse{DOWN_METRIC_INDICATOR}"
TEST_ELLK = f"test_expected_log_likelihood{UP_METRIC_INDICATOR}"
TEST_KL = f"test_kl_divergence{DOWN_METRIC_INDICATOR}"
NOISE_ELLK = f"noise_expected_log_likelihood{DOWN_METRIC_INDICATOR}"
NOISE_KL = f"noise_kl_divergence{DOWN_METRIC_INDICATOR}"

COMPARISON_METRICS = [
    TEST_MLLK,
    TEST_ELBO,
    TEST_MEAN_FIT_MAE,
    TEST_MEAN_FIT_RMSE,
    TEST_VARIANCE_FIT_MAE,
    TEST_VARIANCE_FIT_RMSE,
    TEST_SAMPLE_FIT_MAE,
    TEST_SAMPLE_FIT_RMSE,
    TEST_ELLK,
    TEST_KL,
    NOISE_ELLK,
    NOISE_KL,
]
