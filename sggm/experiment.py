import copy
import json
import pytorch_lightning as pl
import yaml


from argparse import ArgumentParser
from torch.utils.data import DataLoader, random_split

from sggm.definitions import experiment_names, parameters, regression_experiments, regressor_parameters
from sggm.definitions import EXPERIMENT_NAME, EXPERIMENTS_CONFIG

from sggm.regression_model import fit_prior, Regressor


class Experiment:
    def __init__(self, configuration: object):

        # Set dynamically provided configuration attributes
        for k, v in configuration.items():
            self.__dict__[k] = v

        # Otherwise load default values
        self.add_default_params(parameters)

        # No experiment name
        if getattr(self, f'{EXPERIMENT_NAME}', None) is None:
            raise Exception(f'Experiment {json.dumps(self.__dict__)} was not supplied any experiment name')

        # And also load default model specific default values
        # Regression
        if self.experiment_name in regression_experiments:
            self.add_default_params(regressor_parameters)

    def add_default_params(self, params):
        for parameter in params.values():
            if parameter.name not in self.__dict__.keys():
                self.__dict__[parameter.name] = parameter.default

    @property
    def model(self):
        if self.experiment_name in regression_experiments:
            input_dim = 1
            prior_parameters = fit_prior()
            return Regressor(
                input_dim=input_dim, hidden_dim=self.hidden_dim,
                prior_α=prior_parameters[0], prior_β=prior_parameters[1],
                β_out=self.beta_out, eps=self.eps, n_mc_samples=self.n_mc_samples)
        else:
            raise Exception('Experiment does not have a model implemented')
            return

    @property
    def datasets(self):
        return [], [], []


def get_experiments_config(parsed_args):
    if parsed_args.experiments_config:
        with open(parsed_args.experiments_config) as config_file:
            experiments_config = yaml.load(config_file, Loader=yaml.FullLoader)
    else:
        _parsed_args = copy.deepcopy(vars(parsed_args))
        del _parsed_args[f'{EXPERIMENTS_CONFIG}']
        experiments_config = [_parsed_args]
    return experiments_config


def cli_main():

    # ------------
    # Parse args
    # ------------
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument(f'-{EXPERIMENT_NAME}', choices=experiment_names)
    parser.add_argument(f'--{EXPERIMENTS_CONFIG}', type=str)
    # Project wide parameters
    for parameter in parameters.values():
        parser.add_argument(f'--{parameter.name}', default=parameter.default, type=parameter.type)
    args, unknown_args = parser.parse_known_args()

    experiments_config = get_experiments_config(args)

    # ------------
    # Generate experiments
    # ------------
    for experiment_idx, experiment_config in enumerate(experiments_config):

        # Add support for experiment specific arguments
        if f'{EXPERIMENT_NAME}' not in experiment_config.keys():
            raise Exception(f'Experiment {json.dumps(experiment_config)} was not supplied any experiment name')
        else:
            if experiment_config[EXPERIMENT_NAME] in regression_experiments:
                parser = Regressor.add_model_specific_args(parser)

        args, unknown_args = parser.parse_known_args()
        full_experiments_config = get_experiments_config(args)

        experiment = Experiment(full_experiments_config[experiment_idx])

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
        model = experiment.model

        # ------------
        # training
        # ------------
        trainer = pl.Trainer.from_argparse_args(args)
        trainer.fit(model, train_loader, val_loader)

        # ------------
        # testing
        # ------------
        # result = trainer.test(test_dataloaders=test_loader)


if __name__ == '__main__':
    cli_main()
