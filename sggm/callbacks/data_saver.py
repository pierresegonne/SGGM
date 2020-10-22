import pytorch_lightning as pl

from torch import save


class DataSaver(pl.callbacks.Callback):
    def __init__(self, **kwargs):
        super(DataSaver, self).__init__()

    def on_fit_end(self, trainer, pl_module):
        log_dir = trainer.logger.log_dir
        train_dataset = trainer.datamodule.train_dataloader().dataset.dataset
        val_dataset = trainer.datamodule.val_dataloader().dataset.dataset
        save(train_dataset, f"{log_dir}/train_dataset.pkl")
        save(val_dataset, f"{log_dir}/val_dataset.pkl")
