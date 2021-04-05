import numpy as np
import torch

from geoml.nnj import ActivationJacobian, JacType
from torch import nn

log_2_pi = float(torch.log(2 * torch.tensor([np.pi])))


class ShiftLayer(nn.Module):
    def __init__(self, shift_factor):
        super().__init__()
        self.shift_factor = shift_factor

    def forward(self, x):
        return self.shift_factor + x


class NNJ_ShiftLayer(nn.Module, ActivationJacobian):
    def __init__(self, shift_factor):
        super().__init__()
        self.shift_factor = shift_factor

    def forward(self, x, jacobian=False):
        val = self.shift_factor + x

        if jacobian:
            J, _ = self._jacobian(x, val)
            return val, J
        else:
            return val

    def _jacobian(self, x, val):
        J = torch.ones_like(x)
        return J, JacType.DIAG

    def _jac_mul(self, x, val, Jseq, JseqType):
        return Jseq, JseqType
