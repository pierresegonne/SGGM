import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributions as tcd
import torch.nn.functional as F

from argparse import ArgumentParser


class Generator(pl.LightningModule):
    def __init__(self):
        super().__init__()
