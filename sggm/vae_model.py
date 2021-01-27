import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributions as tcd
import torch.nn as nn
import torch.nn.functional as F

from argparse import ArgumentParser

# from pl_bolts.models.autoencoders.components import resnet18_encoder, resnet18_decoder
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
from sggm.definitions import (
    ENCODER_TYPE,
    ENCODER_CONVOLUTIONAL,
    ENCODER_FULLY_CONNECTED,
)
from sggm.definitions import (
    TRAIN_LOSS,
    EVAL_LOSS,
    TEST_LOSS,
)
from sggm.types_ import List, Tensor, Tuple
from sggm.vae_model_helper import batch_flatten, batch_reshape, reduce_int_list


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


def encoder_dense_base(
    input_size: int,
    latent_size: int,
    activation: str,
) -> nn.Module:
    f = activation_function(activation)

    return nn.Sequential(
        nn.Linear(input_size, 512),
        nn.BatchNorm1d(512),
        f,
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        f,
        nn.Linear(256, latent_size),
    )


def decoder_dense_base(
    latent_size: int,
    output_size: int,
    activation: str,
) -> nn.Module:
    f = activation_function(activation)

    return nn.Sequential(
        nn.Linear(latent_size, 256),
        nn.BatchNorm1d(256),
        nn.LeakyReLU(),
        nn.Linear(256, 512),
        nn.BatchNorm1d(512),
        nn.LeakyReLU(),
        nn.Linear(512, output_size),
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
        input_dims: tuple,
        activation: str = F_LEAKY_RELU,
        # encoder_type: str = vae_parameters[ENCODER_TYPE].default,
        latent_dims: Tuple[int] = (10,),
    ):
        super(VanillaVAE, self).__init__()

        self.input_dims = list(input_dims)
        self.input_size = reduce_int_list(self.input_dims)
        self.latent_dims = list(latent_dims)
        self.latent_size = reduce_int_list(self.latent_dims)
        self.activation = activation

        self.lr = 1e-3

        self.encoder_μ = encoder_dense_base(
            self.input_size, self.latent_size, self.activation
        )
        self.encoder_log_var = nn.Sequential(
            encoder_dense_base(self.input_size, self.latent_size, self.activation),
            nn.Softplus(),
        )

        self.decoder_μ = nn.Sequential(
            decoder_dense_base(self.latent_size, self.input_size, self.activation),
            nn.Sigmoid(),
        )
        self.decoder_log_var = nn.Sequential(
            decoder_dense_base(self.latent_size, self.input_size, self.activation),
            nn.Softplus(),
        )

        # Save hparams
        self.save_hyperparameters(
            "activation",
            # "lr",
            "input_dims",
            "latent_dims",
        )

    @staticmethod
    def ellk(p_x_z, x):
        return p_x_z.log_prob(batch_flatten(x))

    @staticmethod
    def kl(q, p, mc_integration: int = 0):
        # Approximate the kl with mc_integration
        if mc_integration >= 1:
            z = q.rsample(torch.Size([mc_integration]))
            return torch.mean(q.log_prob(z) - p.log_prob(z), dim=0)
        return tcd.kl_divergence(q, p)

    @staticmethod
    def elbo(ellk, kl, β_kl: float = 1):
        return ellk - β_kl * kl

    def forward(self, x: Tensor) -> Tensor:
        # Reconstruction
        # To define what I want to use it for ?
        # returns a sample from the decoder, along with its parameters
        x = batch_flatten(x)
        μ_x = self.encoder_μ(x)
        log_var_x = self.encoder_log_var(x)
        z, _, _ = self.sample_latent(μ_x, log_var_x)
        μ_z, log_var_z = self.decoder_μ(z), self.decoder_log_var(z)
        x_hat, p_x = self.sample_generative(μ_z, log_var_z)
        return x_hat, p_x

    def _run_step(self, x):
        # All relevant information for a training step
        # Both latent and generated samples and parameters are returned
        x = batch_flatten(x)
        μ_x = self.encoder_μ(x)
        log_var_x = self.encoder_log_var(x)
        z, q_z_x, p_z = self.sample_latent(μ_x, log_var_x)
        μ_z, log_var_z = self.decoder_μ(z), self.decoder_log_var(z)
        x_hat, p_x_z = self.sample_generative(μ_z, log_var_z)
        x_hat = batch_reshape(x_hat, self.input_dims)
        return x_hat, p_x_z, z, q_z_x, p_z

    def sample_latent(self, mu, log_var):
        std = torch.exp(log_var / 2)
        # batch_shape [batch_shape] event_shape [latent_size]
        q = tcd.Independent(tcd.Normal(mu, std), 1)
        z = q.rsample()  # rsample implies reparametrisation
        p = tcd.Independent(tcd.Normal(torch.zeros_like(z), torch.ones_like(z)), 1)
        return z, q, p

    def sample_generative(self, mu, log_var):
        std = torch.exp(log_var / 2)
        # batch_shape [batch_shape] event_shape [input_size]
        # p = tcd.Independent(tcd.Normal(mu, log_var), 1)
        p = tcd.Independent(tcd.Bernoulli(mu), 1)
        # x = p.rsample()
        x = p.sample()
        return x, p

    def step(self, batch, batch_idx):
        x, y = batch
        x_hat, p_x_z, z, q_z_x, p_z = self._run_step(x)
        # Just learning the mean
        x_hat = batch_reshape(p_x_z.mean, self.input_dims)

        expected_log_likelihood = self.ellk(p_x_z, x)
        kl_divergence = self.kl(q_z_x, p_z)

        loss = -self.elbo(expected_log_likelihood, kl_divergence).mean()

        logs = {
            "ellk": expected_log_likelihood.mean(),
            "kl": kl_divergence.mean(),
            "loss": loss,
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        # self.log(TRAIN_LOSS, loss, on_epoch=True)
        self.log_dict(
            {f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        # Needed for Early stopping?
        self.log(EVAL_LOSS, loss, on_epoch=True)
        self.log_dict(
            {f"val_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False
        )
        # Histogram of weights
        for name, weight in self.named_parameters():
            self.logger.experiment.add_histogram(name, weight, self.current_epoch)
        return loss

    def test_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log(TEST_LOSS, loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # TODO
        # Does self.parameters include children modules parameters?
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        return model_specific_args(vanilla_vae_parameters, parent_parser)


class V3AE(BaseVAE):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        return model_specific_args(v3ae_parameters, parent_parser)
