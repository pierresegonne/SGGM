import argparse

from sggm.analysis.experiment_log import ExperimentLog
from sggm.analysis.parse_results_to_csv import parse_results

from sggm.analysis.cifar_svhn import cifar_svhn_plot
from sggm.analysis.mnist import mnist_plot
from sggm.analysis.sanity_check import sanity_check_plot
from sggm.analysis.toy import toy_plot
from sggm.analysis.toy_2d import toy_2d_plot
from sggm.analysis.uci import uci_plot

from sggm.analysis.utils import str2bool
from sggm.definitions import (
    experiment_names,
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
    #
    CIFAR,
    MNIST,
    MNIST_ND,
    FASHION_MNIST,
    FASHION_MNIST_ND,
    NOT_MNIST,
    SVHN,
    MNIST_ALL,
)
from sggm.regression_model import check_available_methods, MARGINAL

UCI = [
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


def run_analysis(experiment_name, names, model_name, save_dir, **kwargs):
    for name in names:
        experiment_log = ExperimentLog(
            experiment_name, name, model_name=model_name, save_dir=save_dir
        )
        print(
            f"-- Best version: {experiment_log.versions[experiment_log.idx_best_version].version_id}"
        )
        if experiment_name in [SANITY_CHECK]:
            sanity_check_plot(experiment_log, **kwargs)
        elif experiment_name in [TOY, TOY_SHIFTED]:
            toy_plot(experiment_log, **kwargs)
        elif experiment_name == [TOY_2D, TOY_2D_SHIFTED]:
            toy_2d_plot(experiment_log, **kwargs)
        elif experiment_name in UCI:
            uci_plot(experiment_log, **kwargs)
        elif experiment_name in MNIST_ALL:
            mnist_plot(experiment_log, **kwargs)
        elif experiment_name in [CIFAR, SVHN]:
            cifar_svhn_plot(experiment_log, **kwargs)


def add_experiment_args(parser, experiment_name):
    if experiment_name in [SANITY_CHECK, TOY, TOY_SHIFTED, TOY_2D, *UCI]:
        parser.add_argument(
            "--methods",
            type=str,
            default=MARGINAL,
            help="Comma delimited list input, ex 'marginal,posterior'",
        )
    if experiment_name in MNIST_ALL:
        parser.add_argument(
            "--others",
            type=str,
            default=None,
            help="Comma delimited list input, ex 'not_mnist,fashion_mnist'",
        )
    return parser


def parse_experiment_args(args):
    experiment_name = args.experiment_name
    if experiment_name in [SANITY_CHECK, TOY, TOY_SHIFTED, TOY_2D, *UCI]:
        args.methods = [item for item in args.methods.split(",")]
    if experiment_name in MNIST_ALL:
        args.others = (
            [item for item in args.others.split(",")]
            if getattr(args, "others", None) is not None
            else []
        )
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
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="../lightning_logs")
    parser.add_argument(
        "--show_plot",
        type=str2bool,
        required=False,
        const=True,
        nargs="?",
        default=True,
        help="Whether to display the plots while running the analysis",
    )
    args, unknown_args = parser.parse_known_args()

    # Reparse the arguments once the extra arguments have been obtained
    parser = add_experiment_args(parser, args.experiment_name)
    args, unknown_args = parser.parse_known_args()
    args = parse_experiment_args(args)

    run_analysis(**vars(args))
    parse_results(**vars(args))
