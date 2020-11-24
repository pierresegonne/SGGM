import argparse
import os
import pandas as pd

from sggm.definitions import (
    experiment_names,
    COMPARISON_METRICS,
    DOWN_METRIC_INDICATOR,
    UP_METRIC_INDICATOR,
)

MODEL_NAME = "model name"


def compare(experiment_name, names, save_dir):
    # Initialisation
    columns = [MODEL_NAME] + COMPARISON_METRICS
    df = pd.DataFrame(columns=columns)
    mean_df = pd.DataFrame(columns=columns)
    std_df = pd.DataFrame(columns=columns)
    best_df = pd.DataFrame(columns=columns)

    # Gather data
    for name in names:
        dir_path = f"{save_dir}/{experiment_name}/{name}"
        assert os.path.exists(dir_path), f"experiment {dir_path} is not in logs"
        tmp_df = pd.read_csv(f"{dir_path}/results.csv")
        drop_col_labels = [
            col for col in tmp_df.columns if col not in COMPARISON_METRICS
        ]
        tmp_df = tmp_df.drop(labels=drop_col_labels, axis=1)
        tmp_dict = tmp_df[-2:].to_dict(orient="records")
        mean_df = mean_df.append(
            {**{MODEL_NAME: f"{name} (mean)"}, **tmp_dict[0]}, ignore_index=True
        )
        std_df = std_df.append(
            {**{MODEL_NAME: f"{name} (std)"}, **tmp_dict[1]}, ignore_index=True
        )
        best_df = best_df.append(
            {
                **{MODEL_NAME: f"{name} (best)"},
                **{metric: 0 for metric in COMPARISON_METRICS},
            },
            ignore_index=True,
        )

    # Determine best
    for column in COMPARISON_METRICS:
        is_up = UP_METRIC_INDICATOR in column
        is_down = DOWN_METRIC_INDICATOR in column
        if is_up:
            best_df.loc[mean_df[column].idxmax(axis=1), column] = 1
        if is_down:
            best_df.loc[mean_df[column].idxmin(axis=1), column] = 1

    # Aggregate results in single df
    for idx in range(len(names)):  # nbr of names determines # rows
        df = df.append(best_df.iloc[idx], ignore_index=True)
        df = df.append(mean_df.iloc[idx], ignore_index=True)
        df = df.append(std_df.iloc[idx], ignore_index=True)

    # Check existence of comparison folder for experiment
    compare_directory = f"{save_dir}/{experiment_name}/compare/"
    if not os.path.exists(compare_directory):
        os.makedirs(compare_directory)

    # Dump
    df.to_csv(f"{compare_directory}/{'-'.join(names)}.csv", index=False)


def add_experiment_args(parser, experiment_name):
    return parser


def parse_experiment_args(args):
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

    compare(**vars(args))