import argparse
import os
import pandas as pd

from typing import List

from sggm.analysis.utils import str2bool
from sggm.definitions import (
    SHIFTED,
    SHIFTED_SPLIT,
    UCI_ALL,
)


def has_experiment_run(
    save_dir: str, experiment_name: str, run_names: List[str]
) -> bool:
    """ Verifies that the experiment ran for all run names """
    base_path = f"{save_dir}/{experiment_name}"
    for run_name in run_names:
        if not os.path.exists(f"{base_path}/{run_name}"):
            return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--names",
        type=str,
        required=True,
        help="Comma delimited list of run names, ex 'test1,test2'",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        help="Specify model",
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
    parser.add_argument(
        "--shifted_split",
        type=str2bool,
        required=False,
        nargs="?",
        const=True,
        default=False,
        help="Whether to run models on the shifted split experiment or not",
    )
    args, unknown_args = parser.parse_known_args()
    save_dir = "../hpc_lightning_logs" if args.hpc else "../lightning_logs"
    names = [name for name in args.names.split(",")]
    model = args.model

    UCI_EXPERIMENTS = [uci for uci in UCI_ALL if "shifted" not in uci]
    UCI_EXPERIMENTS = (
        UCI_EXPERIMENTS
        if not args.shifted
        else [exp + SHIFTED for exp in UCI_EXPERIMENTS]
    )
    UCI_EXPERIMENTS = (
        UCI_EXPERIMENTS
        if not args.shifted_split
        else [exp + SHIFTED_SPLIT for exp in UCI_EXPERIMENTS]
    )

    if args.run_analysis:
        for i, exp in enumerate(UCI_EXPERIMENTS):
            # Check that data exists
            if has_experiment_run(save_dir, exp, names):

                print(f"\n  -- Running {exp} | {args.names}")

                run_command = f"python run.py --save_dir {save_dir} --experiment_name {exp} --names {args.names}"
                if model is not None:
                    run_command += f" --model_name {model}"
                run_command += " --show_plot 0"
                s_run = os.system(run_command)
                s_compare = os.system(
                    f"python compare.py --save_dir {save_dir} --experiment_name {exp} --names {args.names}"
                )
                # Verify correct execution
                if s_run + s_compare != 0:
                    exit(f"{exp} failed ({s_run},{s_compare})")

                print("  -- OK")

    # Assemble comparison
    # We'll asssume that the name of the csvs don't fallback on _unnamed
    first = True
    if args.hpc is not None:
        for i, exp in enumerate(UCI_EXPERIMENTS):
            if has_experiment_run(save_dir, exp, names):
                # Define overall df
                comparison_filename = (
                    f"{save_dir}/{exp}/compare/{args.names.replace(',', '-')}.csv"
                )
                if first:
                    _df = pd.read_csv(comparison_filename)
                    comp_df = pd.DataFrame(columns=_df.columns)
                    comp_cols = comp_df.columns
                    first = False

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
