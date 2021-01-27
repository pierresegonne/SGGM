import pytorch_lightning as pl
import torchvision

from pytorch_lightning import Callback, Trainer, LightningModule

from sggm.vae_model_helper import batch_reshape

"""
Callback to log generated images
"""


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
            x, y = next(iter(pl_module.test_dataloader()))
            x, y = x.to(pl_module.device), y.to(pl_module.device)
            x_hat, p_x = pl_module(x)

            # Show only mean
            x_hat = batch_reshape(x_hat, pl_module.input_dims)
            x_mean = batch_reshape(p_x.mean, pl_module.input_dims)
            x_var = batch_reshape(p_x.variance, pl_module.input_dims)

            img_list = (
                [x[i] for i in range(self.num_images)]
                + [x_mean[i] for i in range(self.num_images)]
                + [x_var[i] for i in range(self.num_images)]
                + [x_hat[i] for i in range(self.num_images)]
            )
            grid = torchvision.utils.make_grid(img_list, nrow=self.num_images)

            str_title = f"{pl_module.__class__.__name__}_images"
            trainer.logger.experiment.add_image(
                str_title, grid, global_step=trainer.global_step, dataformats="CHW"
            )
