import glob
import matplotlib.pyplot as plt
import os
import pytorch_lightning as pl
import torch

from sggm.styles_ import colours

"""
Saves plots of fit at given interval
Only to be used for toy
"""


class FitSaver(pl.callbacks.Callback):
    def __init__(self, save_every_n_steps: int = 1, **kwargs):
        super(FitSaver, self).__init__()
        self.save_every_n_steps = save_every_n_steps

    def on_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch == 0) | (
            trainer.current_epoch % self.save_every_n_steps == 0
        ):
            x_, y_ = [], []
            mean_ = []
            std_ = []
            with torch.no_grad():
                for (x, y) in pl_module.test_dataloader():
                    mean = pl_module.predictive_mean(x)
                    std = pl_module.predictive_std(x)
                    x_.append(x)
                    y_.append(y)
                    mean_.append(mean)
                    std_.append(std)

            # cat
            x = torch.cat(x_, dim=0).flatten()
            x, idx = torch.sort(x)
            y = torch.cat(y_, dim=0).flatten()[idx]
            mean = torch.cat(mean_, dim=0).flatten()[idx]
            std = torch.cat(std_, dim=0).flatten()[idx]

            # fig
            fig, ax = plt.subplots()
            ax.plot(
                x,
                y,
                "-",
                color="black",
                markersize=2,
            )
            ax.plot(
                x,
                mean,
                "-",
                color=colours["orange"],
                markersize=2,
            )
            ax.fill_between(
                x,
                mean + 1.96 * std,
                mean - 1.96 * std,
                color=colours["orange"],
                alpha=0.3,
            )

            ax.grid(True)
            ax.set_xlim([-5, 15])
            ax.set_ylim([-25, 25])
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$y|x$")

            # save
            log_dir = trainer.logger.log_dir
            save_dir = f"{log_dir}/fit_saves/"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # Check if img already exists, if yes then offset by highest number
            img_name = f"{save_dir}/img{trainer.current_epoch}.png"
            if os.path.exists(img_name) and not hasattr(trainer, "fit_save_offset"):
                offset = max(
                    [
                        int(imgname.split("/")[-1].split(".")[0].split("+")[0][3:])
                        for imgname in glob.glob(f"{save_dir}/*.png")
                    ]
                )
                trainer.fit_save_offset = offset
            if hasattr(trainer, "fit_save_offset"):
                img_name = f"{save_dir}/img{trainer.fit_save_offset}+{trainer.current_epoch}.png"
            plt.savefig(img_name)
            plt.close()
