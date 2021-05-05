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
τ_OOD = "tau_ood"
PRIOR_α = "prior_α"
PRIOR_β = "prior_β"
PRIOR_B = "prior_b"
PRIOR_EPISTEMIC_C = "prior_epistemic_c"
PRIOR_EXTRAPOLATION_X = "prior_extrapolation_x"
OOD_X_GENERATION_METHOD = "ood_x_generation_method"
# Options for OOD_X_GENERATION_METHOD
GAUSSIAN_NOISE = "gaussian_noise"
ADVERSARIAL = "adversarial"
BRUTE_FORCE = "brute_force"
UNIFORM = "uniform"
MEAN_SHIFT = "mean_shift"

OOD_X_GENERATION_AVAILABLE_METHODS = [
    GAUSSIAN_NOISE,
    ADVERSARIAL,
    BRUTE_FORCE,
    UNIFORM,
    MEAN_SHIFT,
]
# Options for OOD_X_GENERATION_AVAILABLE_METHODS
MS_BW_FACTOR = "ms_bw_factor"
MS_KDE_BW_FACTOR = "ms_kde_bw_factor"

OOD_Z_GENERATION_METHOD = "ood_z_generation_method"
# Options for OOD_Z_GENERATION_METHOD
KDE = "kde"
KDE_BANDWIDTH_MULTIPLIER = "kde_bandwidth_multiplier"
GD_PRIOR = "gd_prior"
GD_AGGREGATE_POSTERIOR = "gd_aggregate_posterior"

OOD_Z_GENERATION_AVAILABLE_METHODS = [
    KDE,
    GD_PRIOR,
    GD_AGGREGATE_POSTERIOR,
]

PIG_DL = "pig_dl"
INDUCING_CENTROIDS = "inducing_centroids"
DIGITS = "digits"

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

SPLIT_TRAINING = "split_training"
SPLIT_TRAINING_MODE = "split_training_mode"
# Modes
SPLIT_TRAINING_MSE_MEAN = "split_training_mse_mean"
SPLIT_TRAINING_STD_VV_MEAN = "split_training_std_vv_mean"

DECODER_α_OFFSET = "decoder_α_offset"


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


def none_or_list(value: Any) -> Union[list, None]:
    """Typing for argument being either a list or "None"

    Args:
        value (Any): value whose type will be checked

    Returns:
        Union[str, None]
    """
    if value == "None":
        return None
    assert isinstance(value, list)
    return value


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
    N_TRIALS: Param(N_TRIALS, None, none_or_int),
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
    #
    SPLIT_TRAINING: Param(SPLIT_TRAINING, False, bool),
}
variational_regressor_parameters = {
    β_ELBO: Param(β_ELBO, 0.5, float),
    τ_OOD: Param(τ_OOD, 0.0, float),
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
    #
    SPLIT_TRAINING_MODE: Param(SPLIT_TRAINING_MODE, None, str),
    MS_BW_FACTOR: Param(MS_BW_FACTOR, 1.0, float),
    MS_KDE_BW_FACTOR: Param(MS_KDE_BW_FACTOR, 1.0, float),
}
regression_models = [VARIATIONAL_REGRESSOR]

VANILLA_VAE = "vanilla_vae"
VANILLA_VAE_MANIFOLD = "vanilla_vaem"
VV_VAE = "v3ae"
VV_VAE_MANIFOLD = "v3aem"
vae_parameters = {
    LEARNING_RATE: Param(LEARNING_RATE, 1e-3, float),
    EPS: Param(EPS, 1e-4, float),
    N_MC_SAMPLES: Param(N_MC_SAMPLES, 20, int),
    #
    DIGITS: Param(DIGITS, None, none_or_list),
}
vanilla_vae_parameters = {}
v3ae_parameters = {
    τ_OOD: Param(τ_OOD, 0.0, float),
    OOD_Z_GENERATION_METHOD: Param(
        OOD_Z_GENERATION_METHOD,
        None,
        none_or_str,
        choices=OOD_Z_GENERATION_AVAILABLE_METHODS,
    ),
    PRIOR_α: Param(PRIOR_α, None, float),
    PRIOR_β: Param(PRIOR_β, None, float),
    PRIOR_B: Param(PRIOR_B, None, float),
    PRIOR_EPISTEMIC_C: Param(PRIOR_EPISTEMIC_C, None, float),
    PRIOR_EXTRAPOLATION_X: Param(PRIOR_EXTRAPOLATION_X, None, float),
    KDE_BANDWIDTH_MULTIPLIER: Param(KDE_BANDWIDTH_MULTIPLIER, 10, float),
    DECODER_α_OFFSET: Param(DECODER_α_OFFSET, 0.0, float),
}
generative_models = [VANILLA_VAE, VANILLA_VAE_MANIFOLD, VV_VAE, VV_VAE_MANIFOLD]


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
SHIFTED_SPLIT = "_shifted_split"
# Regression
SANITY_CHECK = "sanity_check"
TOY = "toy"
TOY_SHIFTED = TOY + SHIFTED
TOY_2D = "toy_2d"
TOY_2D_SHIFTED = TOY_2D + SHIFTED
UCI_BOSTON = "uci_boston"
UCI_BOSTON_SHIFTED = UCI_BOSTON + SHIFTED
UCI_BOSTON_SHIFTED_SPLIT = UCI_BOSTON + SHIFTED_SPLIT
UCI_CARBON = "uci_carbon"
UCI_CARBON_SHIFTED = UCI_CARBON + SHIFTED
UCI_CARBON_SHIFTED_SPLIT = UCI_CARBON + SHIFTED_SPLIT
UCI_CCPP = "uci_ccpp"
UCI_CCPP_SHIFTED = UCI_CCPP + SHIFTED
UCI_CCPP_SHIFTED_SPLIT = UCI_CCPP + SHIFTED_SPLIT
UCI_CONCRETE = "uci_concrete"
UCI_CONCRETE_SHIFTED = UCI_CONCRETE + SHIFTED
UCI_CONCRETE_SHIFTED_SPLIT = UCI_CCPP + SHIFTED_SPLIT
UCI_ENERGY = "uci_energy"
UCI_ENERGY_SHIFTED = UCI_ENERGY + SHIFTED
UCI_ENERGY_SHIFTED_SPLIT = UCI_ENERGY + SHIFTED_SPLIT
UCI_KIN8NM = "uci_kin8nm"
UCI_KIN8NM_SHIFTED = UCI_KIN8NM + SHIFTED
UCI_KIN8NM_SHIFTED_SPLIT = UCI_KIN8NM + SHIFTED_SPLIT
UCI_NAVAL = "uci_naval"
UCI_NAVAL_SHIFTED = "uci_naval_shifted"
UCI_NAVAL_SHIFTED_SPLIT = UCI_NAVAL + SHIFTED_SPLIT
UCI_PROTEIN = "uci_protein"
UCI_PROTEIN_SHIFTED = UCI_PROTEIN + SHIFTED
UCI_PROTEIN_SHIFTED_SPLIT = UCI_PROTEIN + SHIFTED_SPLIT
UCI_SUPERCONDUCT = "uci_superconduct"
UCI_SUPERCONDUCT_SHIFTED = UCI_SUPERCONDUCT + SHIFTED
UCI_SUPERCONDUCT_SHIFTED_SPLIT = UCI_SUPERCONDUCT + SHIFTED_SPLIT
UCI_WINE_RED = "uci_wine_red"
UCI_WINE_RED_SHIFTED = UCI_WINE_RED + SHIFTED
UCI_WINE_RED_SHIFTED_SPLIT = UCI_WINE_RED + SHIFTED_SPLIT
UCI_WINE_WHITE = "uci_wine_white"
UCI_WINE_WHITE_SHIFTED = UCI_WINE_WHITE + SHIFTED
UCI_WINE_WHITE_SHIFTED_SPLIT = UCI_WINE_WHITE + SHIFTED_SPLIT
UCI_YACHT = "uci_yacht"
UCI_YACHT_SHIFTED = UCI_YACHT + SHIFTED
UCI_YACHT_SHIFTED_SPLIT = UCI_YACHT + SHIFTED_SPLIT

UCI_ALL = [
    UCI_BOSTON,
    UCI_CARBON,
    UCI_CCPP,
    UCI_CONCRETE,
    UCI_ENERGY,
    UCI_KIN8NM,
    UCI_NAVAL,
    UCI_PROTEIN,
    UCI_SUPERCONDUCT,
    UCI_WINE_RED,
    UCI_WINE_WHITE,
    UCI_YACHT,
]
# Automatically adds the derivatives
UCI_ALL_SHIFTED = [uci + SHIFTED for uci in UCI_ALL]
UCI_ALL_SHIFTED_SPLIT = [uci + SHIFTED_SPLIT for uci in UCI_ALL]
UCI_ALL = UCI_ALL + UCI_ALL_SHIFTED + UCI_ALL_SHIFTED_SPLIT

# Generative
CIFAR = "cifar"
MNIST = "mnist"
MNIST_ND = "mnist_nd"
FASHION_MNIST = "fashion_mnist"
FASHION_MNIST_ND = "fashion_mnist_nd"
NOT_MNIST = "not_mnist"
SVHN = "svhn"
MNIST_ALL = [
    FASHION_MNIST,
    FASHION_MNIST_ND,
    MNIST,
    MNIST_ND,
    NOT_MNIST,
]


experiment_names = (
    [SANITY_CHECK, TOY, TOY_SHIFTED, TOY_2D, TOY_2D_SHIFTED]
    + UCI_ALL
    + MNIST_ALL
    + [CIFAR, SVHN]
)
regression_experiments = [
    SANITY_CHECK,
    TOY,
    TOY_SHIFTED,
    TOY_2D,
    TOY_2D_SHIFTED,
] + UCI_ALL
generative_experiments = [CIFAR, SVHN] + MNIST_ALL

FULLY_CONNECTED = "fully_connected"
CONVOLUTIONAL = "convolutional"
CONV_HIDDEN_DIMS = [32, 64, 128, 256, 512]


def is_shifted_split(experiment_name: str) -> bool:
    return SHIFTED_SPLIT in experiment_name


STAGE_SETUP_SHIFTED_SPLIT = "stage_setup_shifted_split"


def experiments_latent_dims(experiment_name: str) -> tuple:
    if experiment_name == MNIST:
        return (10,)  # Match Nicki and Martin
    if experiment_name == MNIST_ND:
        return (2,)
    elif experiment_name == FASHION_MNIST:
        return (10,)
    elif experiment_name == FASHION_MNIST_ND:
        return (2,)
    elif experiment_name == NOT_MNIST:
        return (10,)
    # Unclear what latent size I should set as default here
    elif experiment_name == CIFAR:
        # TODO change to 128 when proof that everything runs nicely.
        return (128,)
    elif experiment_name == SVHN:
        return (30,)


def experiments_architecture(experiment_name: str) -> str:
    if experiment_name in MNIST_ALL:
        return FULLY_CONNECTED
    elif experiment_name in [CIFAR, SVHN]:
        return CONVOLUTIONAL


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


def experiments_activation_function(experiment_name: str) -> str:
    if experiment_name in [SANITY_CHECK, TOY, TOY_SHIFTED, TOY_2D, TOY_2D_SHIFTED]:
        return F_SIGMOID
    elif experiment_name in UCI_ALL:
        # Match VV
        # return F_RELU
        # Supposed to be best
        return F_ELU
    elif experiment_name in [
        MNIST,
        MNIST_ND,
        FASHION_MNIST,
        FASHION_MNIST_ND,
        NOT_MNIST,
        # Probably needs a convolutional architecture for that
        CIFAR,
        SVHN,
    ]:
        # Match Martin & Nicki
        return F_LEAKY_RELU


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
TEST_OOD_SAMPLE_FIT_MAE = f"test_ood_sample_fit_mae{DOWN_METRIC_INDICATOR}"
TEST_OOD_SAMPLE_FIT_RMSE = f"test_ood_sample_fit_rmse{DOWN_METRIC_INDICATOR}"
TEST_OOD_SAMPLE_MEAN_MSE = f"test_ood_sample_mean_mse{UP_METRIC_INDICATOR}"
TEST_ELLK = f"test_expected_log_likelihood{UP_METRIC_INDICATOR}"
TEST_LLK = f"test_log_likelihood{UP_METRIC_INDICATOR}"
TEST_KL = f"test_kl_divergence{DOWN_METRIC_INDICATOR}"
NOISE_UNCERTAINTY = f"noise_mean_uncertainty{UP_METRIC_INDICATOR}"
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
    TEST_OOD_SAMPLE_FIT_MAE,
    TEST_OOD_SAMPLE_FIT_RMSE,
    TEST_OOD_SAMPLE_MEAN_MSE,
    TEST_ELLK,
    TEST_KL,
    NOISE_UNCERTAINTY,
    NOISE_KL,
]
