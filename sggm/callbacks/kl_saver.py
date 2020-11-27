import matplotlib.pyplot as plt
import os
import pytorch_lightning as pl
import torch

from sggm.styles_ import colours

"Only to be used for toy"


class KLSaver(pl.callbacks.Callback):
    def __init__(self, save_every_n_steps: int = 1, **kwargs):
        super(KLSaver, self).__init__()
        self.save_every_n_steps = save_every_n_steps

    def on_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch == 0) | (
            trainer.current_epoch % self.save_every_n_steps == 0
        ):
            x_ = []
            kl_ = []
            x_out_ = []
            for (x, y) in pl_module.test_dataloader():
                x.requires_grad = True
                μ_x, α_x, β_x = pl_module(x)
                kl_divergence = pl_module.kl(
                    α_x, β_x, pl_module.prior_α, pl_module.prior_β
                )
                x_out = pl_module.ood_x(x, kl=kl_divergence)
                x_.append(x.detach())
                kl_.append(kl_divergence.detach())
                x_out_.append(x_out.detach())

            # cat
            x = torch.cat(x_, dim=0).flatten()
            x, idx = torch.sort(x)
            kl = torch.cat(kl_, dim=0).flatten()[idx]
            x_out = torch.cat(x_out_, dim=0).flatten()

            # fig
            fig, ax = plt.subplots()
            ax.plot(
                x,
                kl,
                "-",
                color=colours["navyBlue"],
            )
            ax.plot(
                x_out,
                torch.zeros_like(x_out),
                "D",
                color=colours["primaryRed"],
                markersize=3,
            )

            ax.grid(True)
            ax.set_xlim([-5, 15])
            ax.set_ylim([-0.1, 1.5])
            ax.set_xlabel("x")
            ax.set_ylabel("KL(x)")

            # save
            log_dir = trainer.logger.log_dir
            save_dir = f"{log_dir}/kl_saves/"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(f"{save_dir}/img{trainer.current_epoch}.png")
            plt.close()
