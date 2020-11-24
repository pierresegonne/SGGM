import argparse

from sggm.analysis.experiment_log import ExperimentLog
from sggm.analysis.parse_results_to_csv import parse_results
from sggm.analysis.toy import toy_plot
from sggm.analysis.toy_2d import toy_2d_plot
from sggm.analysis.uci_yacht import uci_yacht_plot
from sggm.definitions import experiment_names, TOY, TOY_2D, UCI_YACHT
from sggm.regression_model import check_available_methods, MARGINAL


def run_analysis(experiment_name, names, save_dir, **kwargs):
    for name in names:
        experiment_log = ExperimentLog(experiment_name, name, save_dir=save_dir)
        print(f"-- Index of best version: {experiment_log.idx_best_version}")
        if experiment_name == TOY:
            toy_plot(experiment_log, **kwargs)
        elif experiment_name == TOY_2D:
            toy_2d_plot(experiment_log, **kwargs)
        elif experiment_name == UCI_YACHT:
            uci_yacht_plot(experiment_log, **kwargs)


def add_experiment_args(parser, experiment_name):
    if experiment_name in [TOY, TOY_2D, UCI_YACHT]:
        parser.add_argument(
            "--methods",
            type=str,
            default=MARGINAL,
            help="Comma delimited list input, ex 'marginal,posterior'",
        )
    return parser


def parse_experiment_args(args):
    experiment_name = args.experiment_name
    if experiment_name in [TOY, TOY_2D]:
        args.methods = [item for item in args.methods.split(",")]
    args.names = [name for name in args.names.split(",")]
    return args


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_name", type=str, choices=experiment_names, required=True
    )
    parser.add_argument(
        "--names",
        type=str,
        required=True,
        help="Comma delimited list of names, ex 'test1,test2'",
    )
    parser.add_argument("--save_dir", type=str, default="../lightning_logs")
    args, unknown_args = parser.parse_known_args()

    # Reparse the arguments once the extra arguments have been obtained
    parser = add_experiment_args(parser, args.experiment_name)
    args, unknown_args = parser.parse_known_args()
    args = parse_experiment_args(args)

    run_analysis(**vars(args))
    parse_results(**vars(args))
