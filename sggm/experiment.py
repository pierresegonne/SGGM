import copy
import json
import pytorch_lightning as pl
import yaml


from argparse import ArgumentParser
from torch.utils.data import DataLoader, random_split

from sggm.definitions import experiment_names, parameters, regression_experiments
from sggm.definitions import EXPERIMENT_NAME, EXPERIMENTS_CONFIG

from sggm.regression_model import Regressor


class Experiment:
    def __init__(self, configuration: object):

        # Set dynamically provided configuration attributes
        for k, v in configuration.items():
            self.__dict__[k] = v

        # Otherwise load default values
        for parameter in parameters.values():
            if parameter.name not in self.__dict__.keys():
                self.__dict__[parameter.name] = parameter.default

        # No experiment name
        if getattr(self, f'{EXPERIMENT_NAME}', None) is None:
            raise Exception(f'Experiment {json.dumps(self.__dict__)} was not supplied any experiment name')

    @property
    def model(self):
        if self.experiment_name in regression_experiments:
            mdl = Regressor
        return mdl

    @property
    def model_params(self):
        if self.experiment_name in regression_experiments:
            return {}

    @property
    def datasets(self):
        return [], [], []


def cli_main():

    # ------------
    # Parse args
    # ------------
    parser = ArgumentParser()
    parser.add_argument(f'-{EXPERIMENT_NAME}', choices=experiment_names)
    parser.add_argument(f'--{EXPERIMENTS_CONFIG}', type=str)
    for parameter in parameters.values():
        parser.add_argument(f'--{parameter.name}', default=parameter.default, type=parameter.type)
    args = parser.parse_args()

    if args.experiments_config:
        with open(args.experiments_config) as config_file:
            experiments_config = yaml.load(config_file, Loader=yaml.FullLoader)
    else:
        _args = copy.deepcopy(vars(args))
        del _args[f'{EXPERIMENTS_CONFIG}']
        experiments_config = [_args]

    # ------------
    # Generate experiments
    # ------------
    for experiment_config in experiments_config:
        experiment = Experiment(experiment_config)

        # ------------
        # data
        # ------------
        train_dataset, val_dataset, test_dataset = experiment.datasets

        train_loader = DataLoader(train_dataset, batch_size=experiment.batch_size)
        val_loader = DataLoader(val_dataset, batch_size=experiment.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=experiment.batch_size)

        # ------------
        # model
        # ------------
        model = experiment.model(experiment.model_params)

        # ------------
        # training
        # ------------
        trainer = pl.Trainer(
            check_val_every_n_epoch=experiment.check_val_every_n_epoch
        )
        trainer.fit(model, train_loader, val_loader)

        # ------------
        # testing
        # ------------
        # result = trainer.test(test_dataloaders=test_loader)


if __name__ == '__main__':
    cli_main()
