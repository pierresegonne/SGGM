import numpy as np
import torch
import torch.distributions as D

from geoml.nnj import ActivationJacobian, JacType
from torch import nn

from sggm.definitions import (
    ACTIVATION_FUNCTIONS,
    F_ELU,
    F_LEAKY_RELU,
    F_RELU,
    F_SIGMOID,
)

log_2_pi = float(torch.log(2 * torch.tensor([np.pi])))


def get_activation_function(activation: str) -> nn.Module:
    assert (
        activation in ACTIVATION_FUNCTIONS
    ), f"activation_function={activation} is not in {ACTIVATION_FUNCTIONS}"
    if activation == F_ELU:
        f = nn.Tanh()
    if activation == F_LEAKY_RELU:
        f = nn.LeakyReLU()
    elif activation == F_RELU:
        f = nn.ReLU()
    elif activation == F_SIGMOID:
        f = nn.Sigmoid()
    return f


def normalise_grad(grad: torch.Tensor) -> torch.Tensor:
    """Normalise and handle NaNs caused by norm division.

    Args:
        grad (torch.Tensor): Gradient to normalise

    Returns:
        torch.Tensor: Normalised gradient
    """
    normed_grad = grad / torch.linalg.norm(grad, dim=1)[:, None]
    if torch.isnan(normed_grad).any():
        normed_grad = torch.where(
            torch.isnan(normed_grad),
            torch.zeros_like(normed_grad),
            normed_grad,
        )
    return normed_grad


def density_gradient_descent(
    distribution: D.Distribution, x_0: torch.Tensor, params: dict
) -> torch.Tensor:
    N_steps, lr, threshold = params["N_steps"], params["lr"], params["threshold"]

    x_hat = x_0.clone()
    x_hat.requires_grad = True

    print("   PIG gradient descent:", end=" ", flush=True)
    for n in range(N_steps):
        print(f"{n+1}", end=" ")
        with torch.no_grad():
            with torch.set_grad_enabled(True):
                log_prob = distribution.log_prob(x_hat).mean()
                density_grad = torch.autograd.grad(log_prob, x_hat, retain_graph=True)[
                    0
                ]
                normed_density_grad = normalise_grad(density_grad)
                normed_density_grad = torch.where(
                    torch.linalg.norm(density_grad, dim=1)[:, None] < threshold,
                    normed_density_grad,
                    torch.zeros_like(normed_density_grad),
                )
                x_hat = x_hat - lr * normed_density_grad
    print("-> OK")
    x_hat = x_hat.detach()
    return x_hat


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
