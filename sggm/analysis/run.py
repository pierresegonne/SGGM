import argparse

from sggm.analysis.experiment_log import ExperimentLog
from sggm.analysis.toy import toy_plot
from sggm.analysis.toy_2d import toy_2d_plot
from sggm.definitions import experiment_names, TOY, TOY_2D
from sggm.regression_model import check_available_methods, MARGINAL


def run_analysis(experiment_name, name, **kwargs):
    experiment_log = ExperimentLog(experiment_name, name)
    if experiment_name == TOY:
        return toy_plot(experiment_log, **kwargs)
    elif experiment_name == TOY_2D:
        return toy_2d_plot(experiment_log, **kwargs)


def add_experiment_args(parser, experiment_name):
    if experiment_name in [TOY, TOY_2D]:
        parser.add_argument(
            "--methods",
            type=str,
            default=MARGINAL,
            help="Delimited list input, ex 'marginal,posterior'",
        )
    return parser


def parse_experiment_args(args):
    experiment_name = args.experiment_name
    if experiment_name in [TOY, TOY_2D]:
        args.methods = [item for item in args.methods.split(",")]
    return args


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_name", type=str, choices=experiment_names, required=True
    )
    parser.add_argument("--name", type=str, required=True)
    args, unknown_args = parser.parse_known_args()

    # Reparse the arguments once the extra arguments have been obtained
    parser = add_experiment_args(parser, args.experiment_name)
    args, unknown_args = parser.parse_known_args()
    args = parse_experiment_args(args)

    run_analysis(**vars(args))
