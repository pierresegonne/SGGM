import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributions as tcd
import torch.nn as nn
import torch.nn.functional as F

from sggm.types_ import List, Tensor
from argparse import ArgumentParser

from sggm.definitions import generative_parameters
from sggm.definitions import (
    ACTIVATION_FUNCTIONS,
    F_ELU,
    F_LEAKY_RELU,
    F_RELU,
    F_SIGMOID,
)


def activation_function(activation_function_name: str) -> nn.Module:
    assert (
        activation_function in ACTIVATION_FUNCTIONS
    ), f"activation_function={activation_function} is not in {ACTIVATION_FUNCTIONS}"
    if activation_function == F_ELU:
        f = nn.ELU()
    elif activation_function == F_LEAKY_RELU:
        f = nn.LeakyReLU()
    elif activation_function == F_RELU:
        f = nn.ReLU()
    elif activation_function == F_SIGMOID:
        f = nn.Sigmoid()
    return f


def encoder(
    input_dims: int, hidden_dims: List, activation_function_name: str
) -> nn.Module:
    f = activation_function(activation_function_name)
    modules = []
    for h_dim in hidden_dims:
        modules.append(
            nn.Sequential(
                nn.Conv2d(
                    input_dims, out_channels=h_dim, kernel_size=3, stride=2, padding=1
                ),
                nn.BatchNorm2d(h_dim),
                f,
            )
        )
        input_dims = h_dim
    return nn.Sequential(*modules)


def decoder(hidden_dims: List, activation_function_name: str):
    f = activation_function(activation_function_name)
    modules = []
    for i in range(len(hidden_dims) - 1):
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    hidden_dims[i],
                    hidden_dims[i + 1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.BatchNorm2d(hidden_dims[i + 1]),
                f,
            )
        )
    return nn.Sequential(*modules)


def final_layer(hidden_dims: List) -> nn.Module:
    return nn.Sequential(
        nn.ConvTranspose2d(
            hidden_dims[-1],
            hidden_dims[-1],
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        ),
        nn.BatchNorm2d(hidden_dims[-1]),
        nn.LeakyReLU(),
        nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
        nn.Tanh(),
    )


class Generator(pl.LightningModule):
    """
    VAE
    """

    def __init__(
        self,
        input_dims: int = 28 * 28,
        latent_dim: int = 10,
        hidden_dims: List = [32, 64, 128, 256, 512],
        activation_function: str = F_ELU,
    ):
        super(Generator).__init__()

        self.input_dims = input_dims
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.activation_function = activation_function

        self.encoder = encoder(input_dims, hidden_dims, activation_function)
        self.μ_θ = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.log_σ_θ = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        reversed_hidden_dims = hidden_dims.copy()
        reversed_hidden_dims.reverse()

        self.decoder_input = nn.Linear(latent_dim, reversed_hidden_dims[0] * 4)
        self.decoder = decoder(reversed_hidden_dims, activation_function)
        self.final_layer = final_layer(reversed_hidden_dims)

    def encode(self, x: Tensor) -> List[Tensor]:
        enc = self.encoder(x)
        enc = torch.flatten(enc, start_dim=1)

        return [self.μ_θ(enc), self.log_σ_θ(enc)]

    def decode(self, z: Tensor) -> Tensor:
        x = self.decoder_input(z)
        # TODO
        x = x.view()
        x = self.decoder(x)
        x = self.final_layer(x)
        return x

    def reparametrize(self, μ: Tensor, log_σ: Tensor) -> List[Tensor]:
        ε = torch.randn_like(μ)
        return torch.exp(0.5 * log_σ) * ε + μ

    def forward(self, x: Tensor) -> Tensor:
        μ, log_σ = self.encode(x)
        z = self.reparametrize(μ, log_σ)
        x_hat = self.decode(z)
        return [x_hat, μ, log_σ]

    def sample(self, num_samples: int) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim)
        x_hat = self.decode(z)
        return x_hat
    
    def generate(self, x: Tensor) -> Tensor:
        return self.forward(x)[0]

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(
            parents=[parent_parser], add_help=False, conflict_handler="resolve"
        )
        for parameter in generative_parameters.values():
            parser.add_argument(
                f"--{parameter.name}", default=parameter.default, type=parameter.type_
            )
        return parser
