import argparse
import glob
import numpy as np
import pandas as pd
import torch

from sggm.definitions import experiment_names


def parse_results(experiment_name, names, save_dir, **kwargs):
    for name in names:
        # Get all versions
        version_folders = glob.glob(f"{save_dir}/{experiment_name}/{name}/*/")
        assert (
            len(version_folders) > 0
        ), f"No versions found for experiment {save_dir}/{experiment_name}/{name}"

        # Initialise DF
        col = torch.load(f"{version_folders[0]}/results.pkl")[0].keys()
        df = pd.DataFrame(columns=col)

        # Load versions data
        for version_folder in version_folders:
            result = torch.load(f"{version_folder}/results.pkl")[0]
            df = df.append(result, ignore_index=True)

        # Compute mean and std
        mean, std = df.mean(axis=0), df.std(axis=0)

        # Add separation line
        df = df.append({k: None for k in col}, ignore_index=True)

        # Add mean and std
        df = df.append({k: mean[i] for i, k in enumerate(col)}, ignore_index=True)
        df = df.append({k: std[i] for i, k in enumerate(col)}, ignore_index=True)

        df.to_csv(f"{save_dir}/{experiment_name}/{name}/results.csv", index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_name", type=str, choices=experiment_names, required=True
    )
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="../lightning_logs")
    args = parser.parse_args()

    parse_results(args.experiment_name, [args.name], args.save_dir)
