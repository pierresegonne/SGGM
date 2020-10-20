
EXPERIMENT_NAME = 'experiment_name'
EXPERIMENTS_CONFIG = 'experiments_config'

# All parameters for an experiment must be defined here
BATCH_SIZE = 'batch_size'
β_OUT = 'beta_out'
EPS = 'eps'
HIDDEN_DIM = 'hidden_dim'
LEARNING_RATE = 'learning_rate'
NAME = 'name'
N_TRIALS = 'n_trials'
N_MC_SAMPLES = 'n_mc_samples'


class Param:
    def __init__(self, name: str, default: any, type: type):
        self.name = name
        self.default = default
        self.type = type


parameters = {
    BATCH_SIZE: Param(BATCH_SIZE, 32, int),
    HIDDEN_DIM: Param(HIDDEN_DIM, 50, int),
    LEARNING_RATE: Param(LEARNING_RATE, 1e-2, float),
    NAME: Param(NAME, 'unnamed', str),
    N_TRIALS: Param(N_TRIALS, 1, int),
}

# Parameters specific to the Regressor model
regressor_parameters = {
    β_OUT: Param(β_OUT, 1, float),
    EPS: Param(EPS, 1e-10, float),
    N_MC_SAMPLES: Param(N_MC_SAMPLES, 2000, int),
}

# -------------
# Experiment Names
# -------------


TOY = 'toy'
TOY_2D = 'toy_2d'

experiment_names = [
    TOY,
    TOY_2D,
]

regression_experiments = [
    TOY,
    TOY_2D
]
