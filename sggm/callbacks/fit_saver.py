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
            ax.set_xlabel("x")
            ax.set_ylabel("y|x")

            # save
            log_dir = trainer.logger.log_dir
            save_dir = f"{log_dir}/fit_saves/"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(f"{save_dir}/img{trainer.current_epoch}.png")
            plt.close()
