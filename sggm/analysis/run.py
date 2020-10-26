import argparse

from sggm.analysis.experiment_log import ExperimentLog
from sggm.definitions import TOY, experiment_names


def run_analysis(experiment_name, name, **kwargs):
    ExperimentLog(experiment_name, name)
    pass
    # if experiment_name == TOY:
    #     return toy_plot(logger, **kwargs)
    # elif experiment_name == TOY_2D:
    #     return
    # elif experiment_name == UCI_:
    #     return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_name", type=str, choices=experiment_names, required=True
    )
    parser.add_argument("--name", type=str, required=True)
    args = parser.parse_args()

    run_analysis(args.experiment_name, args.name)
