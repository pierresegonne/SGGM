from argparse import ArgumentParser
import copy
from itertools import chain
from typing import List, Tuple

import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

from pl_bolts.models.autoencoders import VAE

from sggm.definitions import (
    model_specific_args,
    vae_parameters,
    vanilla_vae_parameters,
    LEARNING_RATE,
    EPS,
    N_MC_SAMPLES,
)
from sggm.definitions import (
    CONVOLUTIONAL,
    FULLY_CONNECTED,
    RESNET,
    CONV_HIDDEN_DIMS,
    ACTIVATION_FUNCTIONS,
    F_ELU,
    F_LEAKY_RELU,
    F_RELU,
    F_SIGMOID,
)
from sggm.definitions import (
    TRAIN_LOSS,
    EVAL_LOSS,
    TEST_LOSS,
    TEST_ELBO,
    TEST_ELLK,
    TEST_MEAN_FIT_MAE,
    TEST_MEAN_FIT_RMSE,
    TEST_VARIANCE_FIT_MAE,
    TEST_VARIANCE_FIT_RMSE,
    TEST_SAMPLE_FIT_MAE,
    TEST_SAMPLE_FIT_RMSE,
    TEST_OOD_SAMPLE_FIT_MAE,
    TEST_OOD_SAMPLE_FIT_RMSE,
    TEST_OOD_SAMPLE_MEAN_MSE,
)
from sggm.vae_model_helper import (
    batch_flatten,
    batch_reshape,
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
    return nn.Sequential(
        nn.Linear(input_size, 512),
        nn.BatchNorm1d(512),
        activation_function(activation),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        activation_function(activation),
        nn.Linear(256, latent_size),
    )


def encoder_conv_base(
    input_dims: Tuple[int],
    activation: str,
) -> List[nn.Module]:
    input_channels = input_dims[0]

    # Note if we want to handle larger than 32x32 images, leverage input_dims

    modules = [nn.Unflatten(1, input_dims)]
    hidden_dims = CONV_HIDDEN_DIMS

    for h_dim in hidden_dims:
        modules.append(
            nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    out_channels=h_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.BatchNorm2d(h_dim),
                activation_function(activation),
            )
        )
        input_channels = h_dim

    return modules


def decoder_dense_base(
    latent_size: int,
    output_size: int,
    activation: str,
) -> nn.Module:
    return nn.Sequential(
        nn.Linear(latent_size, 256),
        nn.BatchNorm1d(256),
        activation_function(activation),
        nn.Linear(256, 512),
        nn.BatchNorm1d(512),
        activation_function(activation),
        nn.Linear(512, output_size),
    )


def decoder_conv_base(latent_size: int, activation: str) -> List[nn.Module]:
    hidden_dims = copy.deepcopy(CONV_HIDDEN_DIMS)

    # Note * 1 to have 32x32 images

    modules = [
        nn.Linear(latent_size, hidden_dims[-1] * 1),
        nn.Unflatten(1, (hidden_dims[-1], 1, 1)),
    ]

    hidden_dims.reverse()

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
                activation_function(activation),
            )
        )

    return modules


def decoder_conv_final(channels: int, activation: str) -> nn.Module:
    # /!| Still need to apply non-linearity
    return nn.Sequential(
        nn.ConvTranspose2d(
            channels, channels, kernel_size=3, stride=2, padding=1, output_padding=1
        ),
        nn.BatchNorm2d(channels),
        activation_function(activation),
        nn.Conv2d(channels, out_channels=3, kernel_size=3, padding=1),
        nn.Flatten(),
    )


class BaseVAE(pl.LightningModule):
    def __init__(
        self,
        architecture: str,
        input_dims: tuple,
        activation: str,
        latent_dims: Tuple[int],
        learning_rate: float = vae_parameters[LEARNING_RATE].default,
        eps: float = vae_parameters[EPS].default,
        n_mc_samples: int = vae_parameters[N_MC_SAMPLES].default,
    ):
        super().__init__()

        self.architecture = architecture

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
        β = self.β_elbo if train else 1
        return ellk - β * kl

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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, D.Distribution]:
        # Reconstruction
        # Returns generated_samples, decoder
        raise NotImplementedError("Method must be overriden by child VAE model")

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx, stage=VALIDATION)
        # Needed for Early stopping?
        self.log(EVAL_LOSS, loss, on_epoch=True)
        del logs["loss"]
        self.log_dict({f"val_{k}": v for k, v in logs.items()}, on_epoch=True)
        # Histogram of weights
        # for name, weight in self.named_parameters():
        #     self.logger.experiment.add_histogram(name, weight, self.current_epoch)

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
        self.log(TEST_OOD_SAMPLE_FIT_MAE, logs[TEST_OOD_SAMPLE_FIT_MAE], on_epoch=True)
        self.log(
            TEST_OOD_SAMPLE_FIT_RMSE, logs[TEST_OOD_SAMPLE_FIT_RMSE], on_epoch=True
        )
        self.log(
            TEST_OOD_SAMPLE_MEAN_MSE, logs[TEST_OOD_SAMPLE_MEAN_MSE], on_epoch=True
        )
        return loss

    def freeze_but_encoder(self):
        raise NotImplementedError("Method must be overidden by child VAE model")

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        return model_specific_args(vae_parameters, parent_parser)


class VanillaVAE(BaseVAE):
    """
    VAE
    """

    def __init__(
        self,
        architecture: str,
        input_dims: tuple,
        activation: str,
        latent_dims: Tuple[int],
        learning_rate: float = vae_parameters[LEARNING_RATE].default,
        eps: float = vae_parameters[EPS].default,
        n_mc_samples: int = vae_parameters[N_MC_SAMPLES].default,
    ):
        super(VanillaVAE, self).__init__(
            architecture,
            input_dims,
            activation,
            latent_dims=latent_dims,
            learning_rate=learning_rate,
            eps=eps,
            n_mc_samples=n_mc_samples,
        )

        self.β_elbo = 1
        self._switch_to_decoder_var = False
        self._gaussian_decoder = True
        self._bernouilli_decoder = False

        if self.architecture == FULLY_CONNECTED:
            self.encoder_μ = nn.Sequential(
                encoder_dense_base(self.input_size, self.latent_size, self.activation),
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
        elif self.architecture == CONVOLUTIONAL:
            base_encoder = encoder_conv_base(self.input_dims, self.activation)
            self.encoder_μ = nn.Sequential(
                *base_encoder,
                nn.Flatten(),
                nn.Linear(CONV_HIDDEN_DIMS[-1] * 1, self.latent_size),
            )
            self.encoder_std = nn.Sequential(
                *base_encoder,
                nn.Flatten(),
                nn.Linear(CONV_HIDDEN_DIMS[-1] * 1, self.latent_size),
                nn.Softplus(),
            )

            base_decoder = decoder_conv_base(self.latent_size, self.activation)
            self.decoder_μ = nn.Sequential(
                *base_decoder,
                decoder_conv_final(CONV_HIDDEN_DIMS[0], self.activation),
                nn.Sigmoid(),
            )
            self.decoder_std = nn.Sequential(
                *base_decoder,
                decoder_conv_final(CONV_HIDDEN_DIMS[0], self.activation),
                nn.Softplus(),
            )
        elif architecture == RESNET:
            vae = VAE(input_height=96, first_conv=True)
            vae = vae.from_pretrained("stl10-resnet18")

            self.encoder_μ = nn.Sequential(vae.encoder, vae.fc_mu)
            self.encoder_std = nn.Sequential(vae.encoder, vae.fc_var)

            self.decoder_μ = vae.decoder
            self.decoder_std = copy.deepcopy(vae.decoder)

        # %
        self._refit_encoder_mode = False

        # Save hparams
        self.save_hyperparameters(
            "architecture",
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
        if self.architecture == RESNET:
            μ_x = self.encoder_μ(x)
            std_x = self.encoder_std(x)
            std_x = torch.exp(std_x)
            z, _, _ = self.sample_latent(μ_x, std_x)
            μ_z, std_z = self.decoder_μ(z), self.decoder_std(z)
            std_z = torch.exp(std_z)
            x_hat, p_x = self.sample_generative(μ_z, std_z)
            # Seems like the decoder down samples
            x_hat = batch_reshape(x_hat, (3, 32, 32))
        else:
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
        if self.architecture == RESNET:
            # [batch_size, latent_size]
            μ_x = self.encoder_μ(x)
            std_x = torch.exp(self.encoder_std(x))
            # batch_shape [batch_shape] event_shape [latent_size]
            z, q_z_x, p_z = self.sample_latent(μ_x, std_x)
            # [batch_shape, input_size]
            μ_z, std_z = batch_flatten(self.decoder_μ(z)), batch_flatten(
                torch.exp(self.decoder_std(z))
            )
            x_hat, p_x_z = self.sample_generative(μ_z, std_z)
            x_hat = batch_reshape(x_hat, (3, 32, 32))
        else:
            x = batch_flatten(x)
            # [batch_size, latent_size]
            μ_x = self.encoder_μ(x)
            std_x = self.encoder_std(x)
            # batch_shape [batch_shape] event_shape [latent_size]
            z, q_z_x, p_z = self.sample_latent(μ_x, std_x)
            # [batch_shape, input_size]
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
            p = D.Independent(D.Bernoulli(mu, validate_args=False), 1)
            x = p.sample()

        return x, p

    @staticmethod
    def ellk(p_x_z, x):
        x = batch_flatten(x)
        # 1 sample MC integration
        # Seems to work in practice
        return p_x_z.log_prob(x)

    def update_hacks(self):
        if not self._refit_encoder_mode:
            previous_switch = copy.copy(self._switch_to_decoder_var)
            # Switches
            self._switch_to_decoder_var = (
                True
                if (
                    (self.current_epoch > self.trainer.max_epochs / 2)
                    | (self.architecture == RESNET)
                )
                else False
            )
            self._gaussian_decoder = self._switch_to_decoder_var
            self._bernouilli_decoder = not self._switch_to_decoder_var
            # Update optimiser to learn decoder's variance
            # Note: done in training_step
            # Update decoder
            # Note: done with _gaussian_decoder | _bernouilli_decoder
            # Update β_elbo value through annealing
            self.β_elbo = min(1, self.current_epoch / (self.trainer.max_epochs / 2))

            if self._switch_to_decoder_var & (
                previous_switch != self._switch_to_decoder_var
            ):
                for p in self.encoder_μ.parameters():
                    p.requires_grad = False
                for p in self.encoder_std.parameters():
                    p.requires_grad = False
                for p in self.decoder_μ.parameters():
                    p.requires_grad = False

                for m in self.encoder_μ.modules():
                    if isinstance(m, nn.BatchNorm1d) | isinstance(m, nn.BatchNorm2d):
                        m.eval()
                for m in self.encoder_std.modules():
                    if isinstance(m, nn.BatchNorm1d) | isinstance(m, nn.BatchNorm2d):
                        m.eval()
                for m in self.decoder_μ.modules():
                    if isinstance(m, nn.BatchNorm1d) | isinstance(m, nn.BatchNorm2d):
                        m.eval()

        # Handle refitting the encoder
        else:
            for m in self.decoder_μ.modules():
                if isinstance(m, nn.BatchNorm1d) | isinstance(m, nn.BatchNorm2d):
                    m.eval()
            for m in self.decoder_std.modules():
                if isinstance(m, nn.BatchNorm1d) | isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def step(self, batch, batch_idx, stage=None):
        x, y = batch
        x_hat, p_x_z, z, q_z_x, p_z = self._run_step(x)

        if self.architecture == RESNET:
            # Recompute scale factor silences warning, cf doc
            x = F.interpolate(x, scale_factor=1 / 3, recompute_scale_factor=True)

        expected_log_likelihood = self.ellk(p_x_z, x)
        kl_divergence = self.kl(q_z_x, p_z)

        loss = -self.elbo(
            expected_log_likelihood, kl_divergence, train=(stage == TRAINING)
        ).mean()

        if self.architecture == RESNET:
            x_mean = batch_reshape(p_x_z.mean, (3, 32, 32))
        else:
            x_mean = batch_reshape(p_x_z.mean, self.input_dims)
        mean_rmse = torch.sqrt(F.mse_loss(x_mean, x))

        logs = {
            "llk": expected_log_likelihood.sum(),
            "ellk": expected_log_likelihood.mean(),
            "kl": kl_divergence.mean(),
            "loss": loss,
            "mean_rmse": mean_rmse,
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
            if self.architecture == RESNET:
                x_mean = batch_reshape(p_x_z.mean, (3, 32, 32))
            else:
                x_mean = batch_reshape(p_x_z.mean, self.input_dims)
            logs[TEST_MEAN_FIT_MAE] = F.l1_loss(x_mean, x)
            logs[TEST_MEAN_FIT_RMSE] = torch.sqrt(F.mse_loss(x_mean, x))
            # Variance
            if self.architecture == RESNET:
                x_var = batch_reshape(p_x_z.variance, (3, 32, 32))
            else:
                x_var = batch_reshape(p_x_z.variance, self.input_dims)
            empirical_var = (x_mean - x) ** 2
            logs[TEST_VARIANCE_FIT_MAE] = F.l1_loss(x_var, empirical_var)
            logs[TEST_VARIANCE_FIT_RMSE] = torch.sqrt(F.mse_loss(x_var, empirical_var))
            # Samples
            logs[TEST_SAMPLE_FIT_MAE] = F.l1_loss(x_hat, x)
            logs[TEST_SAMPLE_FIT_RMSE] = torch.sqrt(F.mse_loss(x_hat, x))
            # MISC
            logs[TEST_OOD_SAMPLE_FIT_MAE] = 0
            logs[TEST_OOD_SAMPLE_FIT_RMSE] = 0
            logs[TEST_OOD_SAMPLE_MEAN_MSE] = 0

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
        model_opt = torch.optim.Adam(
            chain(
                self.encoder_μ.parameters(),
                self.encoder_std.parameters(),
                self.decoder_μ.parameters(),
            ),
            lr=self.learning_rate,
        )
        decoder_var_opt = torch.optim.Adam(
            self.decoder_std.parameters(), lr=self.learning_rate
        )
        return [model_opt, decoder_var_opt], []

    def freeze_but_encoder(self):
        self._refit_encoder_mode = True
        # Restore gradients of the encoder
        for p in self.encoder_μ.parameters():
            p.requires_grad = True
        for p in self.encoder_std.parameters():
            p.requires_grad = True
        # Freeze gradients of the decoder
        for p in self.decoder_μ.parameters():
            p.requires_grad = False
        for m in self.decoder_μ.modules():
            if isinstance(m, nn.BatchNorm1d) | isinstance(m, nn.BatchNorm2d):
                m.eval()
        for p in self.decoder_std.parameters():
            p.requires_grad = False
        for m in self.decoder_std.modules():
            if isinstance(m, nn.BatchNorm1d) | isinstance(m, nn.BatchNorm2d):
                m.eval()

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        return model_specific_args(vanilla_vae_parameters, parent_parser)
