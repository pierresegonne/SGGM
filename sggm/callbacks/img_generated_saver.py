import io
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torchvision

from PIL import Image
from pytorch_lightning import Callback, Trainer, LightningModule

from sggm.vae_model_helper import batch_reshape

"""
Callback to log generated images
"""


def plot_to_image(fig):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(fig)
    buf.seek(0)
    # Convert PNG buffer to TF image
    # image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = Image.open(buf).convert("RGB")
    image = torchvision.transforms.ToTensor()(image)
    return image


def disable_ticks(ax):
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax.tick_params(axis="y", which="both", left=False, labelleft=False)
    return ax


class IMGGeneratedSaver(pl.callbacks.Callback):
    def __init__(
        self,
        num_images: int = 4,
        save_every_epochs: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.num_images = num_images
        self.save_every_epochs = save_every_epochs

    def on_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if (trainer.current_epoch == 0) | (
            trainer.current_epoch % self.save_every_epochs == 0
        ):
            x, y = next(iter(pl_module.val_dataloader()))
            x, y = x.to(pl_module.device), y.to(pl_module.device)
            pl_module.eval()
            with torch.no_grad():
                # x_hat, p_x = pl_module(x)
                x_hat, p_x_z, λ, q_λ_z, p_λ, z, q_z_x, p_z = pl_module._run_step(x)

                print("\n")
                print("img gen")
                print(pl_module.val_dataloader)
                print(x.shape)
                print(x[0][0][14])
                print(q_z_x.mean[0])
                mean_error = torch.nn.functional.mse_loss(
                    batch_reshape(p_x_z.mean[0], pl_module.input_dims), x
                )
                print("Mean error on val batch", mean_error)

            x_hat = x_hat
            x_mean = batch_reshape(p_x_z.mean, pl_module.input_dims)
            x_var = batch_reshape(p_x_z.variance, pl_module.input_dims)

            fig = plt.figure()
            n_display = 4
            gs = fig.add_gridspec(
                4, n_display, width_ratios=[1] * n_display, height_ratios=[1, 1, 1, 1]
            )
            gs.update(wspace=0, hspace=0)
            for n in range(n_display):
                for k in range(4):
                    ax = plt.subplot(gs[k, n])
                    ax = disable_ticks(ax)
                    # Original
                    if k == 0:
                        ax.imshow(x[n, :][0], cmap="binary", vmin=0, vmax=1)
                    # Mean
                    elif k == 1:
                        ax.imshow(x_mean[n, :][0], cmap="binary", vmin=0, vmax=1)
                    # Variance
                    elif k == 2:
                        ax.imshow(x_var[n, :][0], cmap="binary")
                    # Sample
                    elif k == 3:
                        ax.imshow(x_hat[n, :][0], cmap="binary", vmin=0, vmax=1)

            str_title = f"{pl_module.__class__.__name__}_images"
            trainer.logger.experiment.add_image(
                str_title,
                plot_to_image(fig),
                global_step=trainer.global_step,
                dataformats="CHW",
            )
