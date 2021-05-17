import argparse
import numpy as np
import os
import pathlib
import torch

from codecarbon import EmissionsTracker

from sggm.baselines.utils import timer

from sggm.baselines.gp import gp
from sggm.baselines.john import john
from sggm.baselines.nn import bnn, ensnn, mcdnn, nn


from sggm.definitions import UCI_ALL, TOY, TOY_SHIFTED
from sggm.data import datamodules

ALL_MODELS = ["john", "nn", "mcdnn", "ensnn", "bnn", "gp"]
# experiment_names = UCI_ALL + [TOY, TOY_SHIFTED]
# remove the shifted emmental
experiment_names = [uci for uci in UCI_ALL if uci[-3:] != "ted"] + [TOY, TOY_SHIFTED]


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    gs = parser.add_argument_group("General settings")
    gs.add_argument(
        "--model",
        type=str,
        default=None,
        help="model to use",
        choices=ALL_MODELS,
    )
    gs.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="experiment to run",
        choices=experiment_names,
    )
    gs.add_argument("--seed", type=int, default=1, help="random state of data-split")
    gs.add_argument(
        "--test_split", type=float, default=0.1, help="test set size, as a percentage"
    )
    gs.add_argument("--n_trials", type=int, default=20, help="number of repeatitions")

    ms = parser.add_argument_group("Model specific settings")
    ms.add_argument("--batch_size", type=int, default=512, help="batch size")
    ms.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    ms.add_argument("--iters", type=int, default=10000, help="number of iterations")
    ms.add_argument("--mcmc", type=int, default=500, help="number of mcmc samples")
    ms.add_argument(
        "--inducing", type=int, default=500, help="number of inducing points"
    )
    ms.add_argument(
        "--n_clusters", type=int, default=500, help="number of cluster centers"
    )
    ms.add_argument("--n_models", type=int, default=5, help="number of ensemble")

    # Parse and return
    args = parser.parse_args()
    if args.model is None:
        args.model = ALL_MODELS
    else:
        args.model = [args.model]
    if args.experiment_name is None:
        args.experiment_name = experiment_names
    else:
        args.experiment_name = [args.experiment_name]

    return args


if __name__ == "__main__":
    # baseline model name and dataset
    # if not specified run for all + all

    # Setup dm and extract full datasets to feed in models.
    args = parse_args()

    impact_tracker = EmissionsTracker(
        project_name="sggm_baseline",
        output_dir="./impact_logs/",
        co2_signal_api_token="06297ab81ba8d269",
    )
    impact_tracker.start()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    for model in args.model:
        for experiment_name in args.experiment_name:
            print(
                f"==================== Training model {model} on dataset {experiment_name} ===================="
            )

            # Load data
            dm = datamodules[experiment_name](
                args.batch_size, 0, train_val_split=1.0, test_split=args.test_split
            )
            # if experiment_name in UCI_ALL:

            # elif experiment_name in [TOY, TOY_SHIFTED]:
            #     dm = datamodules[experiment_name]()
            dm.setup()

            # Initialise metrics storage
            log_score, rmse_score = [], []

            # Train multiple models
            T = timer()
            for n_t in range(args.n_trials):
                print(
                    f"==================== Model {n_t + 1}/{args.n_trials} ===================="
                )

                T.begin()
                # Note that execution is not stopped here if error occurs in trial.
                try:
                    logpx, rmse = eval(model)(args, dm)
                except Exception as e:
                    print("encountered error:", e)
                    logpx, rmse = np.nan, np.nan
                T.end()
                log_score.append(logpx)
                rmse_score.append(rmse)

            log_score = np.array(log_score)
            rmse_score = np.array(rmse_score)

            # Save results
            result_folder = f"{pathlib.Path(__file__).parent.absolute()}/results/"
            if result_folder not in os.listdir(
                f"{pathlib.Path(__file__).parent.absolute()}/"
            ):
                os.makedirs(result_folder, exist_ok=True)
            np.savez(
                result_folder + experiment_name + "_" + model,
                log_score=log_score,
                rmse_score=rmse_score,
                timings=np.array(T.timings),
            )

            # Print the results
            print(
                "log(px): {0:.3f} +- {1:.3f}".format(log_score.mean(), log_score.std())
            )
            print(
                "rmse:    {0:.3f} +- {1:.3f}".format(
                    rmse_score.mean(), rmse_score.std()
                )
            )
            T.res()

    emissions = impact_tracker.stop()
    print(f"Emissions: {emissions} kg")
