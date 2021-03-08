import argparse
import os
import pandas as pd

from sggm.analysis.utils import str2bool
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
    parser.add_argument(
        "--run_analysis",
        type=str2bool,
        required=False,
        const=True,
        nargs="?",
        default=True,
        help="Whether to run analysis and comparison of the models",
    )
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
    save_dir = "../hpc_lightning_logs" if args.hpc else "../lightning_logs"

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

    if args.run_analysis:
        for exp in UCI_EXPERIMENTS:
            print(f"\n  -- Running {exp}")

            s_run = os.system(
                f"python run.py --save_dir {save_dir} --experiment_name {exp} --names {args.names} --show_plot 0"
            )
            s_compare = os.system(
                f"python compare.py --save_dir {save_dir} --experiment_name {exp} --names {args.names}"
            )
            # Verify correct execution
            if s_run + s_compare != 0:
                exit(f"{exp} failed ({s_run},{s_compare})")

            print("  -- OK")

    # Assemble comparison
    # We'll asssume that the name of the csvs don't fallback on _unnamed
    if args.hpc is not None:
        for i, exp in enumerate(UCI_EXPERIMENTS):
            # Define overall df
            comparison_filename = (
                f"{save_dir}/{exp}/compare/{args.names.replace(',', '-')}.csv"
            )
            if i == 0:
                _df = pd.read_csv(comparison_filename)
                comp_df = pd.DataFrame(columns=_df.columns)
                comp_cols = comp_df.columns

            _df = pd.read_csv(comparison_filename)

            # Add separation line
            sep_row = {k: None for k in comp_cols}
            comp_df = comp_df.append(sep_row, ignore_index=True)
            sep_row["model name"] = exp
            comp_df = comp_df.append(sep_row, ignore_index=True)
            # Add experiments result
            comp_df = comp_df.append(_df, ignore_index=True)

        comp_df.to_csv(
            f"{save_dir}/uci_comparisons/{args.names.replace(',', '-')}{'-shifted' if args.shifted else ''}.csv",
            index=False,
        )
