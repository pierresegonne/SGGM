import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import seaborn as sns
import torch

from pytorch_lightning import seed_everything
from sklearn.decomposition import PCA
from sggm.data.uci_ccpp.datamodule import (
    UCICCPPDataModule,
    UCICCPPDataModuleShifted,
    COLUMNS as uci_ccpp_columns,
)
from sggm.data.uci_concrete.datamodule import (
    UCIConcreteDataModule,
    UCIConcreteDataModuleShifted,
    COLUMNS as uci_concrete_columns,
)
from sggm.data.uci_wine_red.datamodule import (
    UCIWineRedDataModule,
    UCIWineRedDataModuleShifted,
    COLUMNS as uci_wine_red_columns,
)
from sggm.data.uci_wine_white.datamodule import (
    UCIWineWhiteDataModule,
    UCIWineWhiteDataModuleShifted,
    COLUMNS as uci_wine_white_columns,
)
from sggm.data.uci_yacht.datamodule import (
    UCIYachtDataModule,
    UCIYachtDataModuleShifted,
    COLUMNS as uci_yacht_columns,
)
from sggm.definitions import (
    UCI_CONCRETE,
    UCI_CCPP,
    UCI_SUPERCONDUCT,
    UCI_WINE_RED,
    UCI_WINE_WHITE,
    UCI_YACHT,
)


def main(experiment_name, with_pca=False):

    # Order of plotting
    TEST_FIRST = True

    # Investigate shift effect on pairplot
    SHIFTED = True
    sp_tot = 0.3
    sp_k = 0.0002
    TEST_FIRST = False if SHIFTED else TEST_FIRST

    seed_everything(123)

    # Get correct datamodule
    bs = 10000
    if experiment_name == UCI_CCPP:
        dm = (
            UCICCPPDataModuleShifted(
                bs, 0, shifting_proportion_total=sp_tot, shifting_proportion_k=sp_k
            )
            if SHIFTED
            else UCICCPPDataModule(bs, 0)
        )
        columns = uci_ccpp_columns
    elif experiment_name == UCI_CONCRETE:
        dm = (
            UCIConcreteDataModuleShifted(
                bs, 0, shifting_proportion_total=sp_tot, shifting_proportion_k=sp_k
            )
            if SHIFTED
            else UCIConcreteDataModule(bs, 0)
        )
        columns = uci_concrete_columns
    elif experiment_name == UCI_WINE_RED:
        dm = (
            UCIWineRedDataModuleShifted(
                bs, 0, shifting_proportion_total=sp_tot, shifting_proportion_k=sp_k
            )
            if SHIFTED
            else UCIWineRedDataModule(bs, 0)
        )
        columns = uci_wine_red_columns
    elif experiment_name == UCI_WINE_WHITE:
        dm = (
            UCIWineWhiteDataModuleShifted(
                bs, 0, shifting_proportion_total=sp_tot, shifting_proportion_k=sp_k
            )
            if SHIFTED
            else UCIWineWhiteDataModule(bs, 0)
        )
        columns = uci_wine_white_columns
    elif experiment_name == UCI_YACHT:
        dm = (
            UCIYachtDataModuleShifted(
                bs, 0, shifting_proportion_total=sp_tot, shifting_proportion_k=sp_k
            )
            if SHIFTED
            else UCIYachtDataModule(bs, 0)
        )
        columns = uci_yacht_columns
    dm.setup()

    # Extract data
    train = next(iter(dm.train_dataloader()))
    val = next(iter(dm.val_dataloader()))
    test = next(iter(dm.test_dataloader()))

    print(
        f"N_train={len(dm.train_dataset)}, N_val={len(dm.val_dataset)}, N_test={len(dm.test_dataset)}"
    )

    # 1 = train, 2 = val, 3 = test
    df_columns = columns + ["dataset"]
    df = pd.DataFrame(columns=df_columns)

    if TEST_FIRST:
        dataset_order = [test, val, train]
        dataset_names = ["test", "val", "train"]
    else:
        dataset_order = [train, val, test]
        dataset_names = ["train", "val", "test"]
    for idx_ds, ds in enumerate(dataset_order):
        x, y = ds
        dump = np.concatenate(
            (x.numpy(), y.numpy(), idx_ds * torch.ones_like(y).numpy()), axis=1
        )
        update_df = pd.DataFrame(dump, columns=df_columns)
        df = df.append(update_df, ignore_index=True)

    # correct dataset name
    df["dataset"] = df["dataset"].map({i: v for i, v in enumerate(dataset_names)})

    sns.pairplot(
        df, hue="dataset", palette=sns.color_palette("Set2", len(dataset_names))
    )

    if with_pca:
        pca = PCA(n_components=5)
        pca.fit(df.values[:-1])
        print(pca.explained_variance_ratio_)
        print(pca.singular_values_)
        pca_x = pca.transform(df.values[:-1])

        fig, ax = plt.subplots(1, 1)
        ax.scatter(pca_x[:, 0], pca_x[:, 1])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        choices=[
            UCI_CONCRETE,
            UCI_CCPP,
            UCI_SUPERCONDUCT,
            UCI_WINE_RED,
            UCI_WINE_WHITE,
            UCI_YACHT,
        ],
    )
    args = parser.parse_args()
    main(args.experiment_name)
    plt.show()
