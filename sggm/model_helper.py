import numpy as np
import torch

from torch import nn

log_2_pi = float(torch.log(2 * torch.tensor([np.pi])))


class ShiftLayer(nn.Module):
    def __init__(self, shift_factor):
        super(ShiftLayer, self).__init__()
        self.shift_factor = shift_factor

    def forward(self, x):
        return self.shift_factor + x
