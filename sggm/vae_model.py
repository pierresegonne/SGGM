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
    def __init__(
        self,
        input_dims: tuple,
        activation: str = F_LEAKY_RELU,
        latent_dims: Tuple[int] = (10,),
    ):
        super().__init__()

        self.input_dims = list(input_dims)
        self.input_size = reduce_int_list(self.input_dims)
        self.latent_dims = list(latent_dims)
        self.latent_size = reduce_int_list(self.latent_dims)
        self.activation = activation

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

    def elbo(self, ellk, kl, train: bool = False):
        β = self.β_elbo if train else 1
        return ellk - β * kl

    def sample_latent(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> Tuple[torch.Tensor, tcd.Distribution, tcd.Distribution]:
        # Returns latent_samples, posterior, prior
        raise NotImplementedError("Method must be overriden by child VAE model")

    def sample_generative(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> Tuple[torch.Tensor, tcd.Distribution]:
        # Returns generated_samples, decoder
        raise NotImplementedError("Method must be overriden by child VAE model")

    def forward(self, x: Tensor) -> Tensor:
        # Reconstruction
        # returns a sample from the decoder, along with its parameters
        x = batch_flatten(x)
        μ_x = self.encoder_μ(x)
        log_var_x = self.encoder_log_var(x)
        z, _, _ = self.sample_latent(μ_x, log_var_x)
        μ_z, log_var_z = self.decoder_μ(z), self.decoder_log_var(z)
        x_hat, p_x = self.sample_generative(μ_z, log_var_z)
        return x_hat, p_x

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

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        return model_specific_args(vae_parameters, parent_parser)


class VanillaVAE(BaseVAE):
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
        super(VanillaVAE, self).__init__(
            input_dims, activation=activation, latent_dims=latent_dims
        )

        self.lr = 1e-3
        self.epsilon_std_encoder = 1e-4
        self.β_elbo = 1
        self._switch_to_decoder_var = False
        self._gaussian_decoder = True
        self._bernouilli_decoder = False

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

    @property
    def automatic_optimization(self) -> bool:
        return False

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
        q = tcd.Independent(tcd.Normal(mu, std + self.epsilon_std_encoder), 1)
        z = q.rsample()  # rsample implies reparametrisation
        p = tcd.Independent(tcd.Normal(torch.zeros_like(z), torch.ones_like(z)), 1)
        return z, q, p

    def sample_generative(self, mu, log_var):
        std = torch.exp(log_var / 2)
        # batch_shape [batch_shape] event_shape [input_size]

        if self._gaussian_decoder:
            p = tcd.Independent(tcd.Normal(mu, log_var), 1)
            x = p.rsample()

        if self._bernouilli_decoder:
            p = tcd.Independent(tcd.Bernoulli(mu), 1)
            x = p.sample()

        return x, p

    def update_hacks(self):
        # Switches
        self._switch_to_decoder_var = (
            True if self.current_epoch > self.trainer.max_epochs / 2 else False
        )
        self._gaussian_decoder = self._switch_to_decoder_var
        self._bernouilli_decoder = not self._switch_to_decoder_var
        # Update optimiser to learn decoder's variance
        # Note: done in training_step
        # Update decoder
        # Note: done with _gaussian_decoder | _bernouilli_decoder
        # Update β_elbo value through annealing
        self.β_elbo = min(1, self.current_epoch / (self.trainer.max_epochs / 2))

    def step(self, batch, batch_idx, train=False):

        self.update_hacks()

        x, y = batch
        x_hat, p_x_z, z, q_z_x, p_z = self._run_step(x)
        # Just learning the mean
        x_hat = batch_reshape(p_x_z.mean, self.input_dims)

        expected_log_likelihood = self.ellk(p_x_z, x)
        kl_divergence = self.kl(q_z_x, p_z)

        loss = -self.elbo(expected_log_likelihood, kl_divergence, train=train).mean()

        logs = {
            "ellk": expected_log_likelihood.mean(),
            "kl": kl_divergence.mean(),
            "loss": loss,
        }
        return loss, logs

    def training_step(self, batch, batch_idx, optimizer_idx):
        # print(f"Training step, opt idx: {optimizer_idx}, batch_idx: {batch_idx}")
        (model_opt, decoder_var_opt) = self.optimizers()
        if not self._switch_to_decoder_var:
            opt = model_opt
        else:
            opt = decoder_var_opt
        opt.zero_grad()
        loss, logs = self.step(batch, batch_idx, train=True)

        self.manual_backward(loss, opt)
        opt.step()

        self.log(TRAIN_LOSS, loss, on_epoch=True)
        self.log_dict(
            {f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False
        )
        # print("  ", loss, opt)
        return loss

    def configure_optimizers(self):
        model_opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        decoder_var_opt = torch.optim.Adam(
            self.decoder_log_var.parameters(), lr=self.lr
        )
        return [model_opt, decoder_var_opt], []

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        return model_specific_args(vanilla_vae_parameters, parent_parser)


class V3AE(BaseVAE):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        return model_specific_args(v3ae_parameters, parent_parser)
