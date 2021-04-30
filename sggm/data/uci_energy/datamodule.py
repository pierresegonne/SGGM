# import multiprocessing
# import pytorch_lightning as pl
# import torch

# N_cpus = multiprocessing.cpu_count()


# class DataModule(pl.LightningDataModule):
#     def __init__(
#         self,
#         batch_size: int,
#         n_workers: int,
#     ):
#         super().__init__()

#         self.batch_size = batch_size

#         # Multiprocessing
#         self.n_workers = n_workers if n_workers is not None else N_cpus
#         self.pin_memory = True if self.n_workers > 0 else False
