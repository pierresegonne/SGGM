from typing import Union
import geoml.nnj as nnj
import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

from argparse import ArgumentParser
from itertools import chain
from torch.utils.data import DataLoader, TensorDataset

from geoml import EmbeddedManifold

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
    # %
    OOD_Z_GENERATION_AVAILABLE_METHODS,
    OOD_Z_GENERATION_METHOD,
    KDE,
    PRIOR,
    GD_PRIOR,
    GD_AGGREGATE_POSTERIOR,
    KDE_BANDWIDTH_MULTIPLIER,
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
    TEST_MEAN_FIT_MAE,
    TEST_MEAN_FIT_RMSE,
    TEST_VARIANCE_FIT_MAE,
    TEST_VARIANCE_FIT_RMSE,
    TEST_SAMPLE_FIT_MAE,
    TEST_SAMPLE_FIT_RMSE,
)
from sggm.model_helper import log_2_pi, ShiftLayer
from sggm.types_ import List, Tensor, Tuple
from sggm.vae_model_helper import (
    batch_flatten,
    batch_reshape,
    check_ood_z_generation_method,
    density_gradient_descent,
    reduce_int_list,
)

# stages for steps
TRAINING = "training"
VALIDATION = "validation"
TESTING = "testing"
STAGES = [TRAINING, VALIDATION, TESTING]


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
        return D.kl_divergence(q, p)

    def elbo(self, ellk, kl, train: bool = False):
        β = self.β_elbo if train else 1 / 2
        return 2 * ((1 - β) * ellk - β * kl)

    def sample_latent(
        self, mu: torch.Tensor, std: torch.Tensor
    ) -> Tuple[torch.Tensor, D.Distribution, D.Distribution]:
        # Returns latent_samples, posterior, prior
        raise NotImplementedError("Method must be overriden by child VAE model")

    def sample_generative(
        self, mu: torch.Tensor, std: torch.Tensor
    ) -> Tuple[torch.Tensor, D.Distribution]:
        # Returns generated_samples, decoder
        raise NotImplementedError("Method must be overriden by child VAE model")

    def forward(self, x: Tensor) -> Tuple[torch.Tensor, D.Distribution]:
        # Reconstruction
        # Returns generated_samples, decoder
        raise NotImplementedError("Method must be overriden by child VAE model")

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx, stage=VALIDATION)
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
        loss, logs = self.step(batch, batch_idx, stage=TESTING)
        self.log(TEST_LOSS, logs[TEST_LOSS], on_epoch=True)
        self.log(TEST_ELBO, logs[TEST_ELBO], on_epoch=True)
        self.log(TEST_ELLK, logs[TEST_ELLK], on_epoch=True)
        self.log(TEST_MEAN_FIT_MAE, logs[TEST_MEAN_FIT_MAE], on_epoch=True)
        self.log(TEST_MEAN_FIT_RMSE, logs[TEST_MEAN_FIT_RMSE], on_epoch=True)
        self.log(TEST_VARIANCE_FIT_MAE, logs[TEST_VARIANCE_FIT_MAE], on_epoch=True)
        self.log(TEST_VARIANCE_FIT_RMSE, logs[TEST_VARIANCE_FIT_RMSE], on_epoch=True)
        self.log(TEST_SAMPLE_FIT_MAE, logs[TEST_SAMPLE_FIT_MAE], on_epoch=True)
        self.log(TEST_SAMPLE_FIT_RMSE, logs[TEST_SAMPLE_FIT_RMSE], on_epoch=True)
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
        q = D.Independent(D.Normal(mu, std + self.eps), 1)
        z = q.rsample()  # rsample implies reparametrisation
        p = D.Independent(D.Normal(torch.zeros_like(z), torch.ones_like(z)), 1)
        return z, q, p

    def sample_generative(self, mu, std):
        # batch_shape [batch_shape] event_shape [input_size]
        if self._gaussian_decoder:
            p = D.Independent(D.Normal(mu, std + self.eps), 1)
            x = p.rsample()

        if self._bernouilli_decoder:
            p = D.Independent(D.Bernoulli(mu), 1)
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

    def step(self, batch, batch_idx, stage=None):
        x, y = batch
        x_hat, p_x_z, z, q_z_x, p_z = self._run_step(x)

        expected_log_likelihood = self.ellk(p_x_z, x)
        kl_divergence = self.kl(q_z_x, p_z)

        loss = -self.elbo(
            expected_log_likelihood, kl_divergence, train=(stage == TRAINING)
        ).mean()

        logs = {
            "llk": expected_log_likelihood.sum(),
            "ellk": expected_log_likelihood.mean(),
            "kl": kl_divergence.mean(),
            "loss": loss,
        }

        # Test logs
        if stage == TESTING:
            logs = {}
            logs[TEST_LOSS] = loss
            # ELBO
            logs[TEST_ELBO] = -loss
            # ELLK
            logs[TEST_ELLK] = expected_log_likelihood.mean()
            # MEAN
            x_mean = batch_reshape(p_x_z.mean, self.input_dims)
            logs[TEST_MEAN_FIT_MAE] = F.l1_loss(x_mean, x)
            logs[TEST_MEAN_FIT_RMSE] = torch.sqrt(F.mse_loss(x_mean, x))
            # Variance
            x_var = batch_reshape(p_x_z.variance, self.input_dims)
            empirical_var = (x_mean - x) ** 2
            logs[TEST_VARIANCE_FIT_MAE] = F.l1_loss(x_var, empirical_var)
            logs[TEST_VARIANCE_FIT_RMSE] = torch.sqrt(F.mse_loss(x_var, empirical_var))
            # Samples
            logs[TEST_SAMPLE_FIT_MAE] = F.l1_loss(x_hat, x)
            logs[TEST_SAMPLE_FIT_RMSE] = torch.sqrt(F.mse_loss(x_hat, x))

        return loss, logs

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.update_hacks()
        (model_opt, decoder_var_opt) = self.optimizers()
        if not self._switch_to_decoder_var:
            opt = model_opt
        else:
            opt = decoder_var_opt
        opt.zero_grad()
        loss, logs = self.step(batch, batch_idx, stage=TRAINING)

        self.manual_backward(loss, opt)
        opt.step()

        self.log(TRAIN_LOSS, loss, on_epoch=True)
        self.log_dict(
            {f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False
        )
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
        τ_ood: float = v3ae_parameters[τ_OOD].default,
        eps: float = vae_parameters[EPS].default,
        n_mc_samples: int = vae_parameters[N_MC_SAMPLES].default,
        ood_z_generation_method: str = v3ae_parameters[OOD_Z_GENERATION_METHOD].default,
        kde_bandwidth_multiplier: float = v3ae_parameters[
            KDE_BANDWIDTH_MULTIPLIER
        ].default,
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
        self.ood_z_generation_method = check_ood_z_generation_method(
            ood_z_generation_method
        )

        self.kde_bandwidth_multiplier = kde_bandwidth_multiplier

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
            ShiftLayer(1.0),
        )
        self.decoder_β = nn.Sequential(
            decoder_dense_base(self.latent_size, self.input_size, self.activation),
            nn.Softplus(),
        )

        # set with `set_prior_parameters`
        self.prior_α = None
        self.prior_β = None

        # Save hparams
        self.save_hyperparameters(
            "activation",
            "input_dims",
            "latent_dims",
            "learning_rate",
            "eps",
            "n_mc_samples",
            "τ_ood",
            "ood_z_generation_method",
        )

    # %
    def save_datamodule(self, datamodule: pl.LightningDataModule):
        """ Keep reference of the datamodule on which the model is trained """
        self.dm = datamodule

    def set_prior_parameters(
        self,
        datamodule: pl.LightningDataModule,
        prior_α: Union[int, float, torch.Tensor] = None,
        prior_β: Union[int, float, torch.Tensor] = None,
        min_mode: float = 1e-3,
        max_mode: float = 1e3,
    ):
        """
        Computes adequate prior parameters for the model for the given datamodule.
        Assumes that we can hold the dataset in memory.

        OR
        applies the provided prior parameters
        """
        if (prior_α is None) & (prior_β is None):
            x_train = []
            for idx, batch in enumerate(datamodule.train_dataloader()):
                x, _ = batch
                x_train.append(x)

            # Aggregate the whole dataset
            x_train = torch.cat(x_train, dim=0)
            x_train = torch.reshape(x_train, (-1, *datamodule.dims[1:]))

            x_train_var = x_train.var(dim=0)
            prior_modes = 1 / x_train_var
            prior_modes = torch.maximum(
                prior_modes, min_mode * torch.ones_like(prior_modes)
            )
            prior_modes = torch.minimum(
                prior_modes, max_mode * torch.ones_like(prior_modes)
            )
            self.prior_β = 0.5 * torch.ones_like(prior_modes)
            self.prior_α = 1 + self.prior_β * prior_modes
        elif (prior_α is not None) & (prior_β is not None):
            assert type(prior_α) == type(
                prior_β
            ), "prior_α and prior_β are not of the same type"
            if isinstance(prior_α, float) | isinstance(prior_α, int):
                self.prior_α = prior_α * torch.ones(datamodule.dims)
                self.prior_β = prior_β * torch.ones(datamodule.dims)
            elif isinstance(prior_α, torch.Tensor):
                assert (
                    prior_α.shape == datamodule.dims
                ), "Incorrect dimensions for tensor prior_α"
                assert (
                    prior_β.shape == datamodule.dims
                ), "Incorrect dimensions for tensor prior_β"
                self.prior_α = prior_α
                self.prior_β = prior_β
        else:
            raise ValueError("Incorrect prior values provided, prior_α and prior_β")

    def _setup_pi_dl(self):
        """ Generate a pseudo-input dataloader """
        # Assume that the dataset holds on memory
        print(
            f"\n--- Generating Pseudo-Inputs | {self.ood_z_generation_method}",
            flush=True,
        )
        print("   1. Aggregating DS ...", end=" ", flush=True)
        with torch.no_grad():
            for idx, batch in enumerate(iter(self.dm.train_dataloader())):
                x, _ = batch
                # Actually don't need that, only keep the encoded z, with generation batch by batch
                x_hat, p_x_z, λ, q_λ_z, p_λ, z, q_z_x, p_z = self._run_step(x)
                # Only keep one mc_sample
                z = z[0]
                if idx == 0:
                    agg_z = z
                    agg_q_z_x_mean, agg_q_z_x_stddev = (
                        q_z_x.base_dist.mean,
                        q_z_x.base_dist.stddev,
                    )
                    # Prior is not aggregated as cst
                    p_z_mean, p_z_stddev = p_z.base_dist.mean, p_z.base_dist.stddev
                else:
                    agg_z = torch.cat((agg_z, z), dim=0)
                    agg_q_z_x_mean = torch.cat(
                        (agg_q_z_x_mean, q_z_x.base_dist.mean), dim=0
                    )
                    agg_q_z_x_stddev = torch.cat(
                        (agg_q_z_x_stddev, q_z_x.base_dist.stddev), dim=0
                    )
        print(f"OK | [{agg_z.shape}]", flush=True)

        # two options: pass all in forward pass, or generate pseudo-inputs batch per batch
        q_z_x = D.Independent(D.Normal(agg_q_z_x_mean, agg_q_z_x_stddev), 1)
        p_z_mean, p_z_stddev = (
            p_z_mean[0].repeat(agg_z.shape[0], 1),
            p_z_stddev[0].repeat(agg_z.shape[0], 1),
        )
        p_z = D.Independent(D.Normal(p_z_mean, p_z_stddev), 1)
        z_hat = self.generate_z_out(agg_z, q_z_x, p_z)
        # # Create DL
        self.pig_dl = DataLoader(
            TensorDataset(z_hat), batch_size=self.dm.batch_size, shuffle=True
        )
        print("--- OK\n")

    def generate_z_out(
        self,
        z: torch.Tensor,
        q_z_x: D.Distribution,
        p_z: D.Distribution,
        averaged_std: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        # Only preserve one set of mc samples for efficiency
        assert len(z.shape) == 2, "Incorrect dimensions for z"
        if self.ood_z_generation_method == KDE:
            if averaged_std:
                # Average var accross the BS
                bs = q_z_x.variance.shape[0]
                std = torch.sqrt(torch.mean(q_z_x.variance, dim=0).repeat(bs, 1))
            else:
                std = torch.sqrt(q_z_x.variance)
            # batch_shape [BS] event_shape [event_shape]
            q_out_z_x = D.Independent(
                D.Normal(
                    q_z_x.mean,
                    self.kde_bandwidth_multiplier * std + self.eps,
                ),
                1,
            )
            # [BS, *self.latent_dims]
            z_out = q_out_z_x.rsample((1,)).reshape(*z.shape)
        elif self.ood_z_generation_method == GD_PRIOR:
            # %
            gd_n_steps, gd_lr, gd_threshold = 5, 5e-1, 1
            # batch_shape [] event_shape [event_shape]
            agg_p_z = D.Independent(
                D.Normal(p_z.base_dist.mean[0], p_z.base_dist.stddev[0]),
                1,
            )
            # [BS, *self.latent_dims]
            z_out_start = agg_p_z.sample((z.shape[0],))
            # [BS, *self.latent_dims]
            z_out = density_gradient_descent(
                agg_p_z,
                z_out_start,
                {"N_steps": gd_n_steps, "lr": gd_lr, "threshold": gd_threshold},
            )

        elif self.ood_z_generation_method == GD_AGGREGATE_POSTERIOR:
            # %
            gd_n_steps, gd_lr, gd_threshold = 10, 4e-1, 0.007
            # %
            means, stddevs = (
                q_z_x.base_dist.mean,
                q_z_x.base_dist.stddev,
            )
            agg_q_z_x = D.Independent(D.Normal(means, stddevs), 1)
            mix = D.Categorical(
                torch.ones(
                    means.shape[0],
                )
            )
            agg_q_z_x = D.MixtureSameFamily(mix, agg_q_z_x)
            q_start = D.Independent(
                D.Normal(
                    q_z_x.mean,
                    2 * torch.ones_like(q_z_x.mean),
                ),
                1,
            )
            z_start = q_start.sample((1,)).reshape(*z.shape)
            z_out = density_gradient_descent(
                agg_q_z_x,
                z_start,
                {"N_steps": gd_n_steps, "lr": gd_lr, "threshold": gd_threshold},
            )

        return z_out

    # %

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

        # Prevent beta from collapsing on 0
        β_z = β_z + self.eps

        return z, μ_z, α_z, β_z

    def forward(self, x):
        # Encoding
        x = batch_flatten(x)
        μ_x = self.encoder_μ(x)
        std_x = self.encoder_std(x)
        z, _, _ = self.sample_latent(μ_x, std_x)
        # Decoding
        _, μ_z, α_z, β_z = self.parametrise_z(z)
        x_hat, p_x = self.sample_generative(μ_z, α_z, β_z)
        x_hat = batch_reshape(x_hat, self.input_dims)
        return x_hat, p_x

    def _run_step(self, x):
        # All relevant information for a step
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
        std = std + self.eps
        q = D.Independent(D.Normal(mu, std), 1)
        z = (
            q.rsample(torch.Size([self.n_mc_samples]))
            if mc_samples
            else q.rsample(torch.Size([1]))
        )  # rsample implies reparametrisation
        p = D.Independent(D.Normal(torch.zeros_like(mu), torch.ones_like(std)), 1)
        return z, q, p

    def sample_precision(self, alpha, beta):
        # batch_shape [n_mc_samples, BS] event_shape [input_size]
        # alpha = alpha + self.eps
        # beta = beta + self.eps
        q = D.Independent(D.Gamma(alpha, beta), 1)
        lbd = q.rsample()
        # Reshape prior to match [n_mc_samples, BS, input_size]
        prior_α = self.prior_α.flatten().repeat(alpha.shape[0], alpha.shape[1], 1)
        prior_β = self.prior_β.flatten().repeat(beta.shape[0], beta.shape[1], 1)
        p = D.Independent(
            D.Gamma(
                prior_α,
                prior_β,
            ),
            1,
        )
        return lbd, q, p

    def sample_generative(self, mu, alpha, beta):
        # batch_shape [n_mc_samples, BS] event_shape [input_size]
        # beta = beta + self.eps
        # alpha = alpha + self.eps
        if self._student_t_decoder:
            p = D.Independent(
                D.StudentT(2 * alpha, loc=mu, scale=torch.sqrt(beta / alpha)), 1
            )
            x = p.rsample()

        elif self._bernouilli_decoder:
            # Note the Bernouilli's support is {0, 1} -> not validating args allow to evaluate it on [0, 1]
            # See https://pytorch.org/docs/stable/distributions.html#continuousbernoulli for improvement.
            p = D.Independent(D.Bernoulli(mu, validate_args=False), 1)
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

    def ood_kl(
        self,
        p_λ: D.Distribution,
        z: torch.Tensor,
    ) -> torch.Tensor:

        if self.ood_z_generation_method in OOD_Z_GENERATION_AVAILABLE_METHODS:
            # [n_mc_samples, BS, *self.latent_dims]
            z_out = next(iter(self.pig_dl))[0].type_as(z)
            z_out = z_out[: z.shape[1]]
            z_out = z_out.repeat(self.n_mc_samples, 1, 1)
            # [self.n_mc_samples, BS, self.input_size]
            _, _, α_z_out, β_z_out = self.parametrise_z(z_out)
            # batch_shape [self.n_mc_samples, BS] event_shape [self.input_size]
            q_λ_z_out = D.Independent(D.Gamma(α_z_out, β_z_out), 1)
            # [n_mc_sample, self.input_size]
            kl_divergence_lbd_out = self.kl(q_λ_z_out, p_λ)
            # [self.input_size]
            kl_divergence_lbd_out = torch.mean(kl_divergence_lbd_out, dim=0)
            return kl_divergence_lbd_out
        return torch.zeros((1,))

    def update_hacks(self):
        previous_switch = self._switch_to_decoder_var
        self._switch_to_decoder_var = (
            True if self.current_epoch >= self.trainer.max_epochs / 2 else False
        )
        self._student_t_decoder = self._switch_to_decoder_var
        self._bernouilli_decoder = not self._switch_to_decoder_var
        self.β_elbo = min(1, self.current_epoch / (self.trainer.max_epochs / 2)) / 2
        # Explicitely freeze the gradients of everything but alpha and beta
        if (
            self._switch_to_decoder_var
            and previous_switch != self._switch_to_decoder_var
        ):
            for p in self.encoder_μ.parameters():
                p.requires_grad = False
            for p in self.encoder_std.parameters():
                p.requires_grad = False
            for p in self.decoder_μ.parameters():
                p.requires_grad = False

            self._setup_pi_dl()

    def step(self, batch, batch_idx, stage=None):
        x, _ = batch
        x_hat, p_x_z, λ, q_λ_z, p_λ, z, q_z_x, p_z = self._run_step(x)

        expected_log_likelihood, ellk_lbd, kl_divergence_lbd = self.ellk(
            p_x_z, x, q_λ_z, p_λ
        )
        kl_divergence_z = self.kl(q_z_x, p_z)
        kl_divergence_z = torch.mean(kl_divergence_z, dim=0)

        loss = -self.elbo(
            expected_log_likelihood, kl_divergence_z, train=(stage == TRAINING)
        ).mean()

        # Also verify that we are only training the decoder's variance
        if (
            (stage == TRAINING)
            & (self.ood_z_generation_method is not None)
            & (self._student_t_decoder)
        ):
            # NOTE: beware, for understandability, tau is opposite.
            loss = 2 * (
                (1 - self.τ_ood) * loss + self.τ_ood * self.ood_kl(p_λ, z).mean()
            )

        logs = {
            "llk": expected_log_likelihood.sum(),
            "ellk": expected_log_likelihood.mean(),
            "kl_z": kl_divergence_z.mean(),
            "ellk_lbd": ellk_lbd.mean(),
            "kl_lbd": kl_divergence_lbd.mean(),
            "loss": loss,
        }

        # Test logs
        if stage == TESTING:
            logs = {}
            logs[TEST_LOSS] = loss
            # ELBO
            logs[TEST_ELBO] = -loss
            # ELLK
            logs[TEST_ELLK] = expected_log_likelihood.mean()
            # MEAN
            # Only keep a single z sample
            x_mean = batch_reshape(p_x_z.mean[0], self.input_dims)
            logs[TEST_MEAN_FIT_MAE] = F.l1_loss(x_mean, x)
            logs[TEST_MEAN_FIT_RMSE] = torch.sqrt(F.mse_loss(x_mean, x))
            # Variance
            x_var = batch_reshape(p_x_z.variance[0], self.input_dims)
            empirical_var = (x_mean - x) ** 2
            logs[TEST_VARIANCE_FIT_MAE] = F.l1_loss(x_var, empirical_var)
            logs[TEST_VARIANCE_FIT_RMSE] = torch.sqrt(F.mse_loss(x_var, empirical_var))
            # Samples
            logs[TEST_SAMPLE_FIT_MAE] = F.l1_loss(x_hat, x)
            logs[TEST_SAMPLE_FIT_RMSE] = torch.sqrt(F.mse_loss(x_hat, x))

        return loss, logs

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.update_hacks()
        (model_opt, decoder_var_opt) = self.optimizers()
        if not self._switch_to_decoder_var:
            opt = model_opt
        else:
            opt = decoder_var_opt
        opt.zero_grad()
        loss, logs = self.step(batch, batch_idx, stage=TRAINING)

        self.manual_backward(loss, opt)
        opt.step()

        self.log(TRAIN_LOSS, loss, on_epoch=True)
        self.log_dict(
            {f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False
        )
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


class V3AEm(V3AE, EmbeddedManifold):
    # This will probably fail
    def __init__(self, *args, **kwargs):
        super(V3AEm, self).__init__(*args, **kwargs)

        # Update the decoder networks to allow for Jacobian computation
        # self.decoder_μ = self.nn_to_nnj(self.decoder_μ)
        # self.decoder_α = self.nn_to_nnj(self.decoder_α)
        # self.decoder_β = self.nn_to_nnj(self.decoder_β)

    @staticmethod
    def extract_module_children(mod: nn.Module) -> List[nn.Module]:
        # Extracts all children of a nn module and flattens them
        _children = []
        for child in mod.children():
            _children += V3AEm.extract_module_children(child)
        if len(_children) == 0:
            _children.append(mod)
        return _children

    @staticmethod
    def nn_to_nnj(mod: nn.Module) -> nn.Module:
        # Transform a module with nn layers and activations to nnj module from the geoml package.
        _children = V3AEm.extract_module_children(mod)
        _children_nnj = []
        for child in _children:
            # Layers
            if isinstance(child, nn.Linear):
                bias = True if child.bias is not None else False
                _children_nnj.append(
                    nnj.Linear(child.in_features, child.out_features, bias=bias)
                )
            elif isinstance(child, nn.BatchNorm1d):
                _children_nnj.append(
                    nnj.BatchNorm1d(
                        child.num_features,
                        eps=child.eps,
                        momentum=child.momentum,
                        affine=child.affine,
                        track_running_stats=child.track_running_stats,
                    )
                )
            # Activations
            elif isinstance(child, nn.LeakyReLU):
                _children_nnj.append(nnj.LeakyReLU())
            elif isinstance(child, nn.Sigmoid):
                _children_nnj.append(nnj.Sigmoid())
            elif isinstance(child, nn.Softplus):
                _children_nnj.append(nnj.Softplus())
            else:
                raise NotImplementedError(f"{child} casting to nnj not supported")

    def decoder_jacobian(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, J_μ = self.decoder_μ(z, jacobian=True)
        _, J_α = self.decoder_α(z, jacobian=True)
        _, J_β = self.decoder_β(z, jacobian=True)

        # TODO - WIP but not necessary
        J_σ = 1
        return J_μ, J_σ

    def embed(self, z: torch.Tensor, jacobian=False) -> torch.Tensor:
        is_batched = z.dim() > 2
        if not is_batched:
            z = z.unsqueeze(0)

        # , *self.latent_dims
        assert (
            z.dim() == 3
        ), "Latent codes to embed must be provided as a batch [batch_size, N, *latent_dims]"

        # with n_mc_samples = batch_size
        # [n_mc_samples, BS, *self.latent_dims/self.input_size]
        z, μ_z, α_z, β_z = self.parametrise_z(z)
        # [n_mc_samples, BS, *self.latent_dims/self.input_size]
        σ_z = torch.sqrt(β_z / (α_z - 1))

        # [n_mc_samples, BS, 2 *self.latent_dims/self.input_size]
        embedded = torch.cat((μ_z, σ_z), dim=2)  # BxNx(2D)
        if jacobian:
            J_μ_z, J_σ_z = self.decoder_jacobian(z)
            J = torch.cat((J_μ_z, J_σ_z), dim=2)

        if not is_batched:
            embedded = embedded.squeeze(0)
            if jacobian:
                J = J.squeeze(0)

        if jacobian:
            return embedded, J
        else:
            return embedded
