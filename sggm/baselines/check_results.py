import glob
import numpy as np
from numpy.core.defchararray import mod
import pandas as pd
import pathlib

df_log_score = pd.DataFrame(columns=["experiment_name"])
df_rmse_score = pd.DataFrame(columns=["experiment_name"])
df_timings = pd.DataFrame(columns=["experiment_name"])

PATH = pathlib.Path(__file__).parent.absolute()


def add_experiment_name(df, experiment_name):
    df = df.append(pd.Series(dtype="object"), ignore_index=True)
    df.loc[[len(df) - 1], "experiment_name"] = experiment_name
    return df


def add_model(df, model):
    df[model] = np.NaN
    return df


def fill_value(df, data, model, experiment_name):
    df.loc[
        df.experiment_name == experiment_name, model
    ] = f"{data.mean():.3f}+-{1.96 * data.std():.3f}"
    return df


def add_to_all_df(data, model, experiment_name):
    global df_log_score
    global df_rmse_score
    global df_timings

    log_score = data["log_score"]
    rmse_score = data["rmse_score"]
    timings = data["timings"]

    # if no experiment name, then add corresponding row
    if experiment_name not in df_log_score.experiment_name.values:
        df_log_score = add_experiment_name(df_log_score, experiment_name)
        df_rmse_score = add_experiment_name(df_rmse_score, experiment_name)
        df_timings = add_experiment_name(df_timings, experiment_name)

    # if no model, then add corresponding column
    if model not in df_log_score.columns:
        df_log_score = add_model(df_log_score, model)
        df_rmse_score = add_model(df_rmse_score, model)
        df_timings = add_model(df_timings, model)

    # Add "mean +- 1.96 std" in correct loc
    df_log_score = fill_value(df_log_score, log_score, model, experiment_name)
    df_rmse_score = fill_value(df_rmse_score, rmse_score, model, experiment_name)
    df_timings = fill_value(df_timings, timings, model, experiment_name)


if __name__ == "__main__":

    result_files = glob.glob(f"{PATH}/results/*.npz")

    for result_file in result_files:
        data = np.load(result_file)
        name = result_file.split("/")[-1].split(".")[0]
        model = name.split("_")[-1]
        experiment_name = "_".join(name.split("_")[:-1])
        add_to_all_df(data, model, experiment_name)

    df_log_score.sort_values("experiment_name", inplace=True)
    df_rmse_score.sort_values("experiment_name", inplace=True)
    df_timings.sort_values("experiment_name", inplace=True)


    df_log_score.to_csv(f"{PATH}/results/log_score.csv", index=False)
    df_rmse_score.to_csv(f"{PATH}/results/rmse_score.csv", index=False)
    df_timings.to_csv(f"{PATH}/results/timings_score.csv", index=False)
