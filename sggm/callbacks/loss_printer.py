import pytorch_lightning as pl


class LossPrinter(pl.callbacks.Callback):
    def __init__(self, log_every_n_steps: int = 500, **kwargs):
        super(LossPrinter, self).__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch == 0) | (
            trainer.current_epoch % self.log_every_n_steps == 0
        ):
            # HACK
            # Only works because the last log is saved after log_every_n_steps
            loss = pl_module._results.get_epoch_log_metrics()[
                "train_loss_epoch"
            ].detach()
            print(
                f"[Epoch ({trainer.current_epoch}/{trainer.max_epochs}) Loss: {loss:.4f}]"
            )
