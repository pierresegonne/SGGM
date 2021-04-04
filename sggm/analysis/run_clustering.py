import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from numpy.lib.arraysetops import unique
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier

from sggm.analysis.experiment_log import ExperimentLog
from sggm.definitions import experiment_names, MNIST_ALL
from sggm.analysis.mnist.main import get_dm
from sggm.styles_ import colours, colours_rgb
from sggm.types_ import List, Tuple, Union
from sggm.vae_model import V3AE, VanillaVAE


"""
Run clustering on latent space for VAEs
"""
EUCLIDEAN = "euclidean"
RIEMANNIAN = "riemannian"
METRICS = [EUCLIDEAN, RIEMANNIAN]


def plot_points(
    ax: Axes,
    x: Union[torch.Tensor, np.ndarray],
    y: Union[torch.Tensor, np.ndarray],
    cmap: List[Tuple],
):
    for i_c, c in enumerate(unique(y)):
        ax.plot(
            x[:, 0][y == c],
            x[:, 1][y == c],
            "o",
            markersize=3.5,
            markerfacecolor=(*cmap[i_c], 0.95),
            markeredgewidth=1.2,
            markeredgecolor=(*colours_rgb["white"], 0.5),
            label=f"Class {c}",
        )
    return ax


def clustering_plot_and_metric(experiment_log: ExperimentLog, metric: str):
    assert metric in METRICS

    model = experiment_log.best_version.model

    # Get dm
    bs = 500
    experiment_name = experiment_log.experiment_name
    misc = experiment_log.best_version.misc
    dm = get_dm(experiment_name, misc, bs)

    # Dataset
    # test_dataset = next(iter(dm.val_dataloader()))
    # x_test, y_test = test_dataset

    # Get latent encodings
    z, y = [], []
    with torch.no_grad():
        for idx, batch in enumerate(iter(dm.test_dataloader())):
            x, _y = batch
            if isinstance(model, VanillaVAE):
                _, _, _z, _, _ = model._run_step(x)
            elif isinstance(model, V3AE):
                _, _, _, _, _, _z, _, _ = model._run_step(x)
                _z = _z[0]
            z.append(_z)
            y.append(_y)

    # [len_test_dataset, latent_size]
    z, y = torch.cat(z, dim=0), torch.cat(y)

    # Plots
    colour_names = ["pink", "navyBlue", "yellow"]
    cmap_light = ListedColormap(
        [(*colours_rgb[c_n], 0.4) for c_n in colour_names][: len(misc["digits"])]
    )
    cmap_dark = [colours_rgb[c_n] for c_n in colour_names][: len(misc["digits"])]
    x_mesh, y_mesh = (
        torch.linspace(z[:, 0].min() - 1, z[:, 0].max() + 1, steps=100),
        torch.linspace(z[:, 1].min() - 1, z[:, 1].max() + 1, steps=100),
    )
    x_mesh, y_mesh = torch.meshgrid(x_mesh, y_mesh)
    pos = torch.cat((x_mesh.reshape(-1, 1), y_mesh.reshape(-1, 1)), dim=1)

    # True with NN classifier
    clf = KNeighborsClassifier(n_neighbors=7)
    clf.fit(z, y)
    classes_mesh = clf.predict(pos).reshape(*x_mesh.shape)

    fig, ax = plt.subplots()
    ax.contourf(x_mesh, y_mesh, classes_mesh, cmap=cmap_light)
    ax = plot_points(ax, z, y, cmap_dark)

    # Kmeans
    predicted_classes, predicted_classes_mesh = np.zeros_like(y), np.zeros_like(
        classes_mesh
    )
    if metric == EUCLIDEAN:
        kmeans = KMeans(n_clusters=len(unique(y))).fit(z)
        predicted_classes = kmeans.predict(z)
        predicted_classes_mesh = kmeans.predict(pos).reshape(*x_mesh.shape)
        print(f"[{metric}] F-Score: {f1_score(y, predicted_classes, average='micro')}")

    if metric == RIEMANNIAN:
        pass

    fig, ax = plt.subplots()
    ax.contourf(x_mesh, y_mesh, predicted_classes_mesh, cmap=cmap_light)
    ax = plot_points(ax, z, predicted_classes, cmap_dark)

    plt.show()


# %


def run_clustering(
    experiment_name: str,
    names: List[str],
    metric: str,
    model_name: str,
    save_dir: str,
    **kwargs,
):
    assert experiment_name in MNIST_ALL
    for name in names:
        experiment_log = ExperimentLog(
            experiment_name, name, model_name=model_name, save_dir=save_dir
        )
        clustering_plot_and_metric(experiment_log, metric)


def parse_experiment_args(args):
    args.names = [name for name in args.names.split(",")]
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_name",
        type=str,
        choices=experiment_names,
        required=True,
    )
    parser.add_argument(
        "--names",
        type=str,
        required=True,
        help="Comma delimited list of names, ex 'test1,test2'",
    )
    parser.add_argument("--metric", type=str, choices=METRICS, required=True)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="../lightning_logs")

    args, unknown_args = parser.parse_known_args()
    args = parse_experiment_args(args)

    run_clustering(**vars(args))
