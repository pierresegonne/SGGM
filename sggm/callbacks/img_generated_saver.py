import pytorch_lightning as pl
import torchvision

from pytorch_lightning import Callback, Trainer, LightningModule

"""
Callback to log generated images
"""


class IMGGeneratedSaver(pl.callbacks.Callback):
    def __init__(
        self,
        num_images: int = 3,
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
            x_hat = pl_module(x)
            print(x_hat.shape)
            exit()

            img_list = [x[i] for i in range(self.num_images)] + [
                x_hat[i] for i in range(self.num_images)
            ]
            grid = torchvision.utils.make_grid(img_list, nrow=self.num_images)
            print(grid.shape)

            str_title = f"{pl_module.__class__.__name__}_images"
            trainer.logger.experiment.add_image(
                str_title, grid, global_step=trainer.global_step, dataformats="CHW"
            )
