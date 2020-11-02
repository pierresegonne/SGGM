import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import seaborn as sns
import torch

from sklearn.decomposition import PCA
from sggm.data.uci_yacht.datamodule import UCIYachtDataModule, COLUMNS

# 1 = train, 2 = val, 3 = test
df_columns = COLUMNS + ["dataset"]


def main():
    dm = UCIYachtDataModule(1000, 0)
    dm.setup()
    train = dm.train_dataset.dataset.tensors
    train = (train[0][dm.train_dataset.indices], train[1][dm.train_dataset.indices])
    val = dm.val_dataset.dataset.tensors
    val = (val[0][dm.val_dataset.indices], val[1][dm.val_dataset.indices])
    test = dm.test_dataset.tensors

    df = pd.DataFrame(columns=df_columns)
    for idx_ds, ds in enumerate([train, val, test]):
        x, y = ds
        dump = np.concatenate(
            (x.numpy(), y.numpy(), idx_ds * torch.ones_like(y).numpy()), axis=1
        )
        update_df = pd.DataFrame(dump, columns=df_columns)
        df = df.append(update_df, ignore_index=True)

    sns.pairplot(df, hue="dataset")

    pca = PCA(n_components=5)
    pca.fit(df.values[:-1])
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)
    pca_x = pca.transform(df.values[:-1])

    fig, ax = plt.subplots(1, 1)
    ax.scatter(pca_x[:, 0], pca_x[:, 1])


if __name__ == "__main__":
    main()
    plt.show()
