import argparse
import yaml

from sggm.definitions import EXPERIMENTS_CONFIG

"""
Outputs all the experiment names and individual names of runs that will be executed for a config file.
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(f"--{EXPERIMENTS_CONFIG}", type=str)

    args = parser.parse_args()
    with open(args.experiments_config) as config_file:
        experiments_config = yaml.load(config_file, Loader=yaml.FullLoader)

    all_experiments = {}
    for experiment in experiments_config:
        experiment_name = experiment["experiment_name"]
        if experiment_name in all_experiments:
            all_experiments[experiment_name].append(experiment["name"])
        else:
            all_experiments[experiment_name] = [experiment["name"]]

    print("\n============\nExperiments that will be run:\n")
    for experiment, names in all_experiments.items():
        print(f"-- {experiment}")
        for name in names:
            print(f"   - {name}")
        print("\n")
