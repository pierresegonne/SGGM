import argparse
import glob
import numpy as np
import torch
import yaml

from sggm.definitions import experiment_names
from tomark import Tomark

"""Note that this does not verify that the versions are consistent!
"""


def parse_results(experiment_name, name, save_dir, **kwargs):
    # Get all versions
    version_folders = glob.glob(f"{save_dir}/{experiment_name}/{name}/*/")
    assert (
        len(version_folders) > 0
    ), f"No versions found for experiment {save_dir}/{experiment_name}/{name}"

    # Get hparams
    with open(f"{version_folders[0]}/hparams.yaml") as hparams_file:
        hparams = yaml.load(hparams_file, Loader=yaml.FullLoader)

    # Initialise the results dict
    all_results = {
        k: [] for k in torch.load(f"{version_folders[0]}/results.pkl")[0].keys()
    }

    # Add all individual
    table_individual = []
    for version_folder in version_folders:
        result = torch.load(f"{version_folder}/results.pkl")[0]
        # Condensend gathering
        for k, v in result.items():
            all_results[k].append(v)
        # Table row
        result["version"] = version_folder.split("/")[-2]
        table_individual.append(result)

    # Generate mean and std for all runs
    table_result = {}
    for k, v in all_results.items():
        if k in ["train_loss", "eval_loss", "test_loss", "test_mae", "test_rmse"]:
            table_result[f"mean_{k}"] = np.array(v).mean()
            table_result[f"std_{k}"] = np.array(v).std()
    table_result = [table_result]

    # Save in readable format
    with open(f"{save_dir}/{experiment_name}/{name}/results.md", "w") as results_file:
        results_file.write(f"# Results - {experiment_name} - {name}\n\n")

        results_file.write('![Visual output](_.png "Visual output")\n\n')

        results_file.write("## Hparams\n\n")
        table_hparams = Tomark.table([hparams])
        results_file.write(table_hparams)
        results_file.write("\n\n")

        results_file.write("## Aggregated Runs\n\n")
        table_result = Tomark.table(table_result)
        results_file.write(table_result)
        results_file.write("\n\n")

        results_file.write("## Individual Runs\n\n")
        table_individual = Tomark.table(table_individual)
        results_file.write(table_individual)
        results_file.write("\n\n")

        results_file.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_name", type=str, choices=experiment_names, required=True
    )
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="../lightning_logs")
    args = parser.parse_args()

    parse_results(args.experiment_name, args.name, args.save_dir)
