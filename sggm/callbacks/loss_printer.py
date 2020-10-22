import pytorch_lightning as pl


class LossPrinter(pl.callbacks.Callback):
    def __init__(self, print_every_n_epoch: int = 500, **kwargs):
        super(LossPrinter, self).__init__()
        self.print_every_n_epoch = print_every_n_epoch

    def on_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch == 0) | (
            trainer.current_epoch % self.print_every_n_epoch == 0
        ):
            print("epoch end")
            print(trainer.datamodule)
            print(trainer.logger.log_dir)