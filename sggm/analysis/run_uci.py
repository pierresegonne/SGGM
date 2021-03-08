import argparse
import os

from sggm.analysis.utils import none_to_str, str2bool
from sggm.definitions import (
    UCI_CCPP,
    UCI_CONCRETE,
    UCI_SUPERCONDUCT,
    UCI_WINE_RED,
    UCI_WINE_WHITE,
    UCI_YACHT,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--names",
        type=str,
        required=True,
        help="Comma delimited list of names, ex 'test1,test2'",
    )
    parser.add_argument(
        "--hpc",
        type=str2bool,
        required=False,
        const=True,
        nargs="?",
        default=True,
        help="Whether to run models from the hpc_lightning_logs folder or not",
    )
    # Careful this is true no matter what value is given in the arg

    parser.add_argument(
        "--shifted",
        type=str2bool,
        required=False,
        nargs="?",
        const=True,
        default=False,
        help="Whether to run models on the shifted experiment or not",
    )
    args, unknown_args = parser.parse_known_args()
    save_dir = "--save_dir ../hpc_lightning_logs" if args.hpc else None

    UCI_EXPERIMENTS = [
        UCI_CCPP,
        UCI_CONCRETE,
        UCI_SUPERCONDUCT,
        UCI_WINE_RED,
        UCI_WINE_WHITE,
        UCI_YACHT,
    ]
    UCI_EXPERIMENTS = (
        UCI_EXPERIMENTS
        if not args.shifted
        else [f"{exp}_shifted" for exp in UCI_EXPERIMENTS]
    )

    for exp in UCI_EXPERIMENTS:
        print(f"\n  -- Running {exp}")

        s_run = os.system(
            f"python run.py {none_to_str(save_dir)} --experiment_name {exp} --names {args.names} --show_plot 0"
        )
        s_compare = os.system(
            f"python compare.py {none_to_str(save_dir)} --experiment_name {exp} --names {args.names}"
        )
        # Verify correct execution
        if s_run + s_compare != 0:
            exit(f"{exp} failed ({s_run},{s_compare})")

        print("  -- OK")
