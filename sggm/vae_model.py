import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributions as tcd
import torch.nn as nn
import torch.nn.functional as F

from sggm.types_ import List, Tensor
from argparse import ArgumentParser
from pl_bolts.models.autoencoders.components import resnet18_encoder, resnet18_decoder

from sggm.definitions import (
    model_specific_args,
    vae_parameters,
    vanilla_vae_parameters,
    v3ae_parameters,
)
from sggm.definitions import (
    ACTIVATION_FUNCTIONS,
    F_ELU,
    F_LEAKY_RELU,
    F_RELU,
    F_SIGMOID,
)


def activation_function(activation_function_name: str) -> nn.Module:
    assert (
        activation_function_name in ACTIVATION_FUNCTIONS
    ), f"activation_function={activation_function} is not in {ACTIVATION_FUNCTIONS}"
    if activation_function_name == F_ELU:
        f = nn.ELU()
    elif activation_function_name == F_LEAKY_RELU:
        f = nn.LeakyReLU()
    elif activation_function_name == F_RELU:
        f = nn.ReLU()
    elif activation_function_name == F_SIGMOID:
        f = nn.Sigmoid()
    return f


def encoder(
    input_dims: int, hidden_dims: List, activation_function_name: str
) -> nn.Module:
    f = activation_function(activation_function_name)
    modules = []
    for h_dim in hidden_dims:
        print(h_dim)
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


class BaseVAE(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        return model_specific_args(vae_parameters, parent_parser)


class VanillaVAE(pl.LightningModule):
    """
    VAE
    """

    def __init__(
        self,
        input_dim: int = 28,
        latent_dim: int = 10,
        enc_out_dim: int = 128,
        activation_function_name: str = F_ELU,
    ):
        super(VanillaVAE, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.enc_out_dim = enc_out_dim
        self.activation_function_name = activation_function_name

        self.lr = 1e-4

        # Default
        first_conv = False
        maxpool1 = False

        self.encoder = resnet18_encoder(first_conv, maxpool1)
        self.μ_θ = nn.Linear(self.enc_out_dim, self.latent_dim)
        self.log_var = nn.Linear(self.enc_out_dim, self.latent_dim)

        self.decoder = resnet18_decoder(
            self.latent_dim, self.input_dim, first_conv, maxpool1
        )

    @staticmethod
    def ellk(x, x_hat):
        return -F.mse_loss(x_hat, x, reduction="mean")

    @staticmethod
    def kl(q, p):
        return torch.mean(tcd.kl_divergence(q, p))

    @staticmethod
    def elbo(ellk, kl):
        return ellk - kl

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        μ = self.μ_θ(x)
        log_var = self.log_var(x)
        p, q, z = self.sample(μ, log_var)
        return self.decoder(z)

    def _run_step(self, x):
        x = self.encoder(x)
        mu = self.μ_θ(x)
        log_var = self.log_var(x)
        p, q, z = self.sample(mu, log_var)
        return z, self.decoder(z), p, q

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = tcd.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = tcd.Normal(mu, std)
        z = q.rsample()  # rsample implies reparametrisation
        return p, q, z

    def step(self, batch, batch_idx):
        x, y = batch
        z, x_hat, p, q = self._run_step(x)

        expected_log_likelihood = self.ellk(x, x_hat)
        kl_divergence = self.kl(q, p)

        loss = -self.elbo(expected_log_likelihood, kl_divergence)

        logs = {
            "recon_loss": expected_log_likelihood,
            "kl": kl_divergence,
            "loss": loss,
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict(
            {f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

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


class V3AE(BaseVAE):
    pass
