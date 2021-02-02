import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributions as tcd
import torch.nn as nn
import torch.nn.functional as F

from argparse import ArgumentParser
from itertools import chain

from sggm.definitions import (
    model_specific_args,
    vae_parameters,
    vanilla_vae_parameters,
    v3ae_parameters,
    LEARNING_RATE,
    PRIOR_α,
    PRIOR_β,
    τ_OOD,
    EPS,
    N_MC_SAMPLES,
    OOD_Z_GENERATION_METHOD,
    KDE,
    PRIOR,
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
    TEST_ELBO,
    TEST_ELLK,
    TEST_KL,
    TEST_LLK,
)
from sggm.model_helper import log_2_pi, ShiftLayer
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
        activation: str,
        latent_dims: Tuple[int],
        learning_rate: float = vae_parameters[LEARNING_RATE].default,
        eps: float = vae_parameters[EPS].default,
        n_mc_samples: int = vae_parameters[N_MC_SAMPLES].default,
    ):
        super().__init__()

        self.input_dims = list(input_dims)
        self.input_size = reduce_int_list(self.input_dims)
        self.latent_dims = list(latent_dims)
        self.latent_size = reduce_int_list(self.latent_dims)
        self.activation = activation

        self.example_input_array = torch.rand((16, *self.input_dims))

        self.learning_rate = learning_rate

        self.eps = eps
        self.n_mc_samples = n_mc_samples

    def kl(self, q, p, mc_integration: bool = False):
        # Approximate the kl with mc_integration
        if mc_integration:
            z = q.rsample(torch.Size([self.n_mc_samples]))
            return torch.mean(q.log_prob(z) - p.log_prob(z), dim=0)
        return tcd.kl_divergence(q, p)

    def elbo(self, ellk, kl, train: bool = False):
        β = self.β_elbo if train else 1 / 2
        return (1 - β) * ellk - β * kl

    def sample_latent(
        self, mu: torch.Tensor, std: torch.Tensor
    ) -> Tuple[torch.Tensor, tcd.Distribution, tcd.Distribution]:
        # Returns latent_samples, posterior, prior
        raise NotImplementedError("Method must be overriden by child VAE model")

    def sample_generative(
        self, mu: torch.Tensor, std: torch.Tensor
    ) -> Tuple[torch.Tensor, tcd.Distribution]:
        # Returns generated_samples, decoder
        raise NotImplementedError("Method must be overriden by child VAE model")

    def forward(self, x: Tensor) -> Tuple[torch.Tensor, tcd.Distribution]:
        # Reconstruction
        # Returns generated_samples, decoder
        raise NotImplementedError("Method must be overriden by child VAE model")

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
        activation: str,
        latent_dims: Tuple[int],
        learning_rate: float = vae_parameters[LEARNING_RATE].default,
        eps: float = vae_parameters[EPS].default,
        n_mc_samples: int = vae_parameters[N_MC_SAMPLES].default,
        # encoder_type: str = vae_parameters[ENCODER_TYPE].default,
    ):
        super(VanillaVAE, self).__init__(
            input_dims,
            activation,
            latent_dims=latent_dims,
            learning_rate=learning_rate,
            eps=eps,
            n_mc_samples=n_mc_samples,
        )

        self.β_elbo = 1 / 2
        self._switch_to_decoder_var = False
        self._gaussian_decoder = True
        self._bernouilli_decoder = False

        self.encoder_μ = encoder_dense_base(
            self.input_size, self.latent_size, self.activation
        )
        self.encoder_std = nn.Sequential(
            encoder_dense_base(self.input_size, self.latent_size, self.activation),
            nn.Softplus(),
        )

        self.decoder_μ = nn.Sequential(
            decoder_dense_base(self.latent_size, self.input_size, self.activation),
            nn.Sigmoid(),
        )
        self.decoder_std = nn.Sequential(
            decoder_dense_base(self.latent_size, self.input_size, self.activation),
            nn.Softplus(),
        )

        # Save hparams
        self.save_hyperparameters(
            "activation",
            "input_dims",
            "latent_dims",
            "learning_rate",
            "eps",
            "n_mc_samples",
        )

    @property
    def automatic_optimization(self) -> bool:
        return False

    def forward(self, x):
        x = batch_flatten(x)
        μ_x = self.encoder_μ(x)
        std_x = self.encoder_std(x)
        z, _, _ = self.sample_latent(μ_x, std_x)
        μ_z, std_z = self.decoder_μ(z), self.decoder_std(z)
        x_hat, p_x = self.sample_generative(μ_z, std_z)
        x_hat = batch_reshape(x_hat, self.input_dims)
        return x_hat, p_x

    def _run_step(self, x):
        # All relevant information for a training step
        # Both latent and generated samples and parameters are returned
        x = batch_flatten(x)
        μ_x = self.encoder_μ(x)
        std_x = self.encoder_std(x)
        z, q_z_x, p_z = self.sample_latent(μ_x, std_x)
        μ_z, std_z = self.decoder_μ(z), self.decoder_std(z)
        x_hat, p_x_z = self.sample_generative(μ_z, std_z)
        x_hat = batch_reshape(x_hat, self.input_dims)
        return x_hat, p_x_z, z, q_z_x, p_z

    def sample_latent(self, mu, std):
        # batch_shape [batch_shape] event_shape [latent_size]
        q = tcd.Independent(tcd.Normal(mu, std + self.eps), 1)
        z = q.rsample()  # rsample implies reparametrisation
        p = tcd.Independent(tcd.Normal(torch.zeros_like(z), torch.ones_like(z)), 1)
        return z, q, p

    def sample_generative(self, mu, std):
        # batch_shape [batch_shape] event_shape [input_size]
        if self._gaussian_decoder:
            p = tcd.Independent(tcd.Normal(mu, std + self.eps), 1)
            x = p.rsample()

        if self._bernouilli_decoder:
            p = tcd.Independent(tcd.Bernoulli(mu), 1)
            x = p.sample()

        return x, p

    @staticmethod
    def ellk(p_x_z, x):
        x = batch_flatten(x)
        # 1 sample MC integration
        # Seems to work in practice
        return p_x_z.log_prob(x)

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
        self.β_elbo = min(1, self.current_epoch / (self.trainer.max_epochs / 2)) / 2

    def step(self, batch, batch_idx, train=False):

        if self.training:
            self.update_hacks()

        x, y = batch
        x_hat, p_x_z, z, q_z_x, p_z = self._run_step(x)

        expected_log_likelihood = self.ellk(p_x_z, x)
        kl_divergence = self.kl(q_z_x, p_z)

        loss = -self.elbo(expected_log_likelihood, kl_divergence, train=train).mean()

        logs = {
            "llk": expected_log_likelihood.sum(),
            "ellk": expected_log_likelihood.mean(),
            "kl": kl_divergence.mean(),
            "loss": loss,
        }
        return loss, logs

    def training_step(self, batch, batch_idx, optimizer_idx):
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
        return loss

    def test_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log(TEST_LOSS, loss, on_epoch=True)
        self.log(TEST_ELBO, -loss, on_epoch=True)
        self.log(TEST_ELLK, logs["ellk"], on_epoch=True)
        self.log(TEST_KL, logs["kl"], on_epoch=True)
        self.log(TEST_LLK, logs["llk"], on_epoch=True)
        return loss

    def configure_optimizers(self):
        model_opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        decoder_var_opt = torch.optim.Adam(
            self.decoder_std.parameters(), lr=self.learning_rate
        )
        return [model_opt, decoder_var_opt], []

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        return model_specific_args(vanilla_vae_parameters, parent_parser)


class V3AE(BaseVAE):
    """
    V3AE
    """

    def __init__(
        self,
        input_dims: tuple,
        activation: str,
        latent_dims: Tuple[int],
        learning_rate: float = vae_parameters[LEARNING_RATE].default,
        prior_α: float = v3ae_parameters[PRIOR_α].default,
        prior_β: float = v3ae_parameters[PRIOR_β].default,
        τ_ood: float = v3ae_parameters[τ_OOD].default,
        eps: float = vae_parameters[EPS].default,
        n_mc_samples: int = vae_parameters[N_MC_SAMPLES].default,
        ood_z_generation_method: str = v3ae_parameters[OOD_Z_GENERATION_METHOD].default,
        # encoder_type: str = vae_parameters[ENCODER_TYPE].default,
    ):
        super(V3AE, self).__init__(
            input_dims,
            activation,
            latent_dims=latent_dims,
            learning_rate=learning_rate,
            eps=eps,
            n_mc_samples=n_mc_samples,
        )

        self.β_elbo = 1 / 2

        self.τ_ood = τ_ood
        self.ood_z_generation_method = ood_z_generation_method

        self.kde_bandwidth_multiplier = 10

        self._switch_to_decoder_var = False
        self._student_t_decoder = True
        self._bernouilli_decoder = False

        self.encoder_μ = encoder_dense_base(
            self.input_size, self.latent_size, self.activation
        )
        self.encoder_std = nn.Sequential(
            encoder_dense_base(self.input_size, self.latent_size, self.activation),
            nn.Softplus(),
        )

        self.decoder_μ = nn.Sequential(
            decoder_dense_base(self.latent_size, self.input_size, self.activation),
            nn.Sigmoid(),
        )
        self.decoder_α = nn.Sequential(
            decoder_dense_base(self.latent_size, self.input_size, self.activation),
            nn.Softplus(),
            ShiftLayer(1),
        )
        self.decoder_β = nn.Sequential(
            decoder_dense_base(self.latent_size, self.input_size, self.activation),
            nn.Softplus(),
        )

        self.prior_α = prior_α
        self.prior_β = prior_β

        # Save hparams
        self.save_hyperparameters(
            "activation",
            "input_dims",
            "latent_dims",
            "learning_rate",
            "eps",
            "n_mc_samples",
            "prior_α",
            "prior_β",
            "τ_ood",
            "ood_z_generation_method",
        )

    def parametrise_z(self, z):
        bs = z.shape[1]
        # [self.n_mc_samples, BS, *self.latent_dims] -> [self.n_mc_samples * BS, *self.latent_dims]
        z = torch.reshape(z, [-1, *self.latent_dims])
        μ_z = self.decoder_μ(z)
        α_z = self.decoder_α(z)
        β_z = self.decoder_β(z)
        # [self.n_mc_samples * BS, self.input_size] -> [self.n_mc_samples, BS, self.input_size]
        μ_z = torch.reshape(μ_z, [-1, bs, self.input_size])
        α_z = torch.reshape(α_z, [-1, bs, self.input_size])
        β_z = torch.reshape(β_z, [-1, bs, self.input_size])
        # [self.n_mc_samples * BS, *self.latent_dims] -> [self.n_mc_samples, BS, *self.latent_dims]
        z = torch.reshape(z, [-1, bs, *self.latent_dims])
        return z, μ_z, α_z, β_z

    def forward(self, x):
        x = batch_flatten(x)
        μ_x = self.encoder_μ(x)
        std_x = self.encoder_std(x)
        z, _, _ = self.sample_latent(μ_x, std_x)
        _, μ_z, α_z, β_z = self.parametrise_z(z)
        x_hat, p_x = self.sample_generative(μ_z, α_z, β_z)
        x_hat = batch_reshape(x_hat, self.input_dims)
        return x_hat, p_x

    def _run_step(self, x):
        # All relevant information for a training step
        # Both latent and generated samples and parameters are returned
        x = batch_flatten(x)
        μ_x = self.encoder_μ(x)
        std_x = self.encoder_std(x)
        z, q_z_x, p_z = self.sample_latent(
            μ_x, std_x, mc_samples=self._student_t_decoder
        )
        # [self.n_mc_samples, BS, *self.latent_dims/self.input_size]
        z, μ_z, α_z, β_z = self.parametrise_z(z)
        # [self.n_mc_samples, BS, self.input_size]
        λ, q_λ_z, p_λ = self.sample_precision(α_z, β_z)
        # [BS, self.input_size], [n_mc_sample, BS, self.input_size]
        x_hat, p_x_z = self.sample_generative(μ_z, α_z, β_z)
        # [BS, *self.input_dims]
        x_hat = batch_reshape(x_hat, self.input_dims)
        return x_hat, p_x_z, λ, q_λ_z, p_λ, z, q_z_x, p_z

    def sample_latent(self, mu, std, mc_samples: bool = False):
        # batch_shape [batch_shape] event_shape [latent_size]
        q = tcd.Independent(tcd.Normal(mu, std + self.eps), 1)
        z = (
            q.rsample(torch.Size([self.n_mc_samples]))
            if mc_samples
            else q.rsample(torch.Size([1]))
        )  # rsample implies reparametrisation
        p = tcd.Independent(tcd.Normal(torch.zeros_like(z), torch.ones_like(z)), 1)
        return z, q, p

    def sample_precision(self, alpha, beta):
        # batch_shape [n_mc_samples, BS] event_shape [input_size]
        q = tcd.Independent(tcd.Gamma(alpha, beta), 1)
        lbd = q.rsample()
        p = tcd.Independent(
            tcd.Gamma(
                self.prior_α * torch.ones_like(alpha),
                self.prior_β * torch.ones_like(beta),
            ),
            1,
        )
        return lbd, q, p

    def sample_generative(self, mu, alpha, beta):
        # batch_shape [n_mc_samples, BS] event_shape [input_size]
        beta = beta + self.eps
        if self._student_t_decoder:
            p = tcd.Independent(
                tcd.StudentT(2 * alpha, loc=mu, scale=torch.sqrt(beta / alpha)), 1
            )
            x = p.rsample()

        elif self._bernouilli_decoder:
            p = tcd.Independent(tcd.Bernoulli(mu), 1)
            x = p.sample()

        # first dim is num of latent z samples, only keep the reconstructed from the first draw
        x = x[0]

        return x, p

    def ellk(self, p_x_z, x, q_λ_z, p_λ):
        x = batch_flatten(x)
        if self._student_t_decoder:
            # [n_mc_sample, BS, self.input_size]
            μ_z = p_x_z.mean
            α_z = p_x_z.base_dist.df / 2
            β_z = (p_x_z.base_dist.scale ** 2) * α_z

            expected_log_lambda = torch.digamma(α_z) - torch.log(β_z)
            expected_lambda = α_z / β_z
            # [n_mc_sample, self.input_size]
            ellk_lbd = torch.sum(
                (1 / 2)
                * (expected_log_lambda - log_2_pi - expected_lambda * ((x - μ_z) ** 2)),
                dim=2,
            )
            # [self.input_size]
            ellk_lbd = torch.mean(ellk_lbd, dim=0)

            # [n_mc_sample, self.input_size]
            kl_divergence_lbd = self.kl(q_λ_z, p_λ)
            # [self.input_size]
            kl_divergence_lbd = torch.mean(kl_divergence_lbd, dim=0)
            return (
                ellk_lbd - kl_divergence_lbd,
                ellk_lbd,
                kl_divergence_lbd,
            )

        elif self._bernouilli_decoder:
            return (
                p_x_z.log_prob(x),
                torch.zeros((x.shape[0], 1)),
                torch.zeros((x.shape[0], 1)),
            )

    def ood_kl(self, p_λ, q_z_x):

        if self.ood_z_generation_method == KDE:
            # Average var accross the BS
            std = torch.sqrt(torch.mean(q_z_x.variance), dim=1)
            print(std.shape)
            exit()
            # batch_shape [BS] event_shape [event_shape]
            q_out_z_x = tcd.Independent(
                tcd.Normal(
                    q_z_x.mean,
                    self.kde_bandwidth_multiplier * torch.sqrt(q_z_x.variance)
                    + self.eps,
                ),
                1,
            )
            # [n_mc_samples, BS, *self.latent_dims]
            z_out = q_out_z_x.rsample(torch.Size([self.n_mc_samples]))
            # [self.n_mc_samples, BS, self.input_size]
            _, _, α_z_out, β_z_out = self.parametrise_z(z_out)
            # batch_shape [self.n_mc_samples, BS] event_shape [self.input_size]
            q_λ_z_out = tcd.Independent(tcd.Gamma(α_z_out, β_z_out), 1)
            # [n_mc_sample, self.input_size]
            kl_divergence_lbd_out = self.kl(q_λ_z_out, p_λ)
            # [self.input_size]
            kl_divergence_lbd_out = torch.mean(kl_divergence_lbd_out, dim=0)
            return kl_divergence_lbd_out
        return 0

    def update_hacks(self):
        self._switch_to_decoder_var = (
            True if self.current_epoch > self.trainer.max_epochs / 2 else False
        )
        self._student_t_decoder = self._switch_to_decoder_var
        self._bernouilli_decoder = not self._switch_to_decoder_var
        self.β_elbo = min(1, self.current_epoch / (self.trainer.max_epochs / 2)) / 2

    def step(self, batch, batch_idx, train=False):
        if self.training:
            self.update_hacks()

        x, y = batch
        x_hat, p_x_z, λ, q_λ_z, p_λ, z, q_z_x, p_z = self._run_step(x)

        expected_log_likelihood, ellk_lbd, kl_divergence_lbd = self.ellk(
            p_x_z, x, q_λ_z, p_λ
        )
        kl_divergence_z = self.kl(q_z_x, p_z)
        kl_divergence_z = torch.mean(kl_divergence_z, dim=0)

        loss = -self.elbo(expected_log_likelihood, kl_divergence_z, train=train).mean()

        # Also verify that we are only training the decoder's variance
        if (
            (train)
            & (self.ood_z_generation_method is not None)
            & (self._student_t_decoder)
        ):
            # NOTE: beware, for understandability, tau is opposite.
            loss = (1 - self.τ_ood) * loss + self.τ_ood * self.ood_kl(p_λ, q_z_x).mean()

        logs = {
            "llk": expected_log_likelihood.sum(),
            "ellk": expected_log_likelihood.mean(),
            "kl_z": kl_divergence_z.mean(),
            "ellk_lbd": ellk_lbd.mean(),
            "kl_lbd": kl_divergence_lbd.mean(),
            "loss": loss,
        }
        return loss, logs

    def training_step(self, batch, batch_idx, optimizer_idx):
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
        return loss

    def test_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log(TEST_LOSS, loss, on_epoch=True)
        self.log(TEST_ELBO, -loss, on_epoch=True)
        self.log(TEST_ELLK, logs["ellk"], on_epoch=True)
        self.log(TEST_KL, logs["kl_z"], on_epoch=True)
        self.log(TEST_LLK, logs["llk"], on_epoch=True)
        return loss

    def configure_optimizers(self):
        model_opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        decoder_var_opt = torch.optim.Adam(
            chain(self.decoder_α.parameters(), self.decoder_β.parameters()),
            lr=self.learning_rate,
        )
        return [model_opt, decoder_var_opt], []

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        return model_specific_args(v3ae_parameters, parent_parser)
