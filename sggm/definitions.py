
EXPERIMENT_NAME = 'experiment_name'
EXPERIMENTS_CONFIG = 'experiments_config'

# All parameters for an experiment must be defined here
BATCH_SIZE = 'batch_size'
CHECK_VAL_EVERY_N_EPOCH = 'check_val_every_n_epoch'
HIDDEN_DIM = 'hidden_dim'
LEARNING_RATE = 'learning_rate'
MAX_EPOCHS = 'max_epochs'
NAME = 'name'
N_TRIALS = 'n_trials'


class Param:
    def __init__(self, name: str, default: any, type: type):
        self.name = name
        self.default = default
        self.type = type


parameters = {
    BATCH_SIZE: Param(BATCH_SIZE, 32, int),
    CHECK_VAL_EVERY_N_EPOCH: Param(CHECK_VAL_EVERY_N_EPOCH, 10, int),
    HIDDEN_DIM: Param(HIDDEN_DIM, 50, int),
    LEARNING_RATE: Param(LEARNING_RATE, 1e-2, float),
    NAME: Param(NAME, 'unnamed', str),
    N_TRIALS: Param(N_TRIALS, 1, int),
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
