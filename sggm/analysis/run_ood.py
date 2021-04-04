import argparse
import os

from sggm.analysis.experiment_log import ExperimentLog
from sggm.analysis.mnist import mnist_plot
from sggm.analysis.mnist.main import get_dm
from sggm.definitions import (
    MNIST_ALL,
    experiment_names,
)


"""
Run the analysis plots on OOD inputs
"""


def run_analysis(experiment_name, run_on, names, model_name, save_dir, **kwargs):
    for name in names:
        experiment_log = ExperimentLog(
            experiment_name, name, model_name=model_name, save_dir=save_dir
        )
        save_folder = f"{experiment_log.save_dir}/{experiment_log.experiment_name}/{experiment_log.name}/{run_on}/"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        if experiment_name in MNIST_ALL:
            dm = get_dm(run_on, experiment_log.best_version.misc, 500)
            mnist_plot(experiment_log, dm=dm, save_folder=save_folder, **kwargs)


def parse_experiment_args(args):
    args.names = [name for name in args.names.split(",")]
    return args


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_name", type=str, choices=experiment_names, required=True
    )
    parser.add_argument("--run_on", type=str, choices=experiment_names, required=True)
    parser.add_argument(
        "--names",
        type=str,
        required=True,
        help="Comma delimited list of names, ex 'test1,test2'",
    )
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="../lightning_logs")

    args, unknown_args = parser.parse_known_args()
    args = parse_experiment_args(args)

    run_analysis(**vars(args))
