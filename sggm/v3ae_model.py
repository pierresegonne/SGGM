from argparse import ArgumentParser
import copy
from itertools import chain
from typing import Tuple, Union

import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F


from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset

from sggm.definitions import (
    model_specific_args,
    vae_parameters,
    v3ae_parameters,
    LEARNING_RATE,
    τ_OOD,
    EPS,
    N_MC_SAMPLES,
    PRIOR_B,
    PRIOR_EPISTEMIC_C,
    PRIOR_EXTRAPOLATION_X,
    PRIOR_EXTRAPOLATION_AVAILABLE_MODES,
    PRIOR_EXTRAPOLATION_MODE,
    PRIOR_EXTRAPOLATION_ON_CENTROIDS,
    PRIOR_EXTRAPOLATION_ON_PI,
    # %
    OOD_Z_GENERATION_AVAILABLE_METHODS,
    OOD_Z_GENERATION_METHOD,
    KDE,
    GD_PRIOR,
    GD_AGGREGATE_POSTERIOR,
    KDE_BANDWIDTH_MULTIPLIER,
    DECODER_α_OFFSET,
)
from sggm.definitions import (
    CONVOLUTIONAL,
    FULLY_CONNECTED,
    RESNET,  # TODO
    CONV_HIDDEN_DIMS,
)
from sggm.definitions import (
    TRAIN_LOSS,
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
from sggm.model_helper import log_2_pi, ShiftLayer
from sggm.model_helper import density_gradient_descent
from sggm.vae_model import (
    BaseVAE,
    decoder_conv_base,
    decoder_conv_final,
    decoder_dense_base,
    encoder_conv_base,
    encoder_dense_base,
    TESTING,
    TRAINING,
)
from sggm.vae_model_helper import (
    batch_flatten,
    batch_reshape,
    check_ood_z_generation_method,
    locscale_sigmoid,
)


class V3AE(BaseVAE):
    """
    V3AE
    """

    def __init__(
        self,
        architecture: str,
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
        prior_b: Union[float, None] = v3ae_parameters[PRIOR_B].default,
        prior_epistemic_c: Union[float, None] = v3ae_parameters[
            PRIOR_EPISTEMIC_C
        ].default,
        prior_extrapolation_x: Union[float, None] = v3ae_parameters[
            PRIOR_EXTRAPOLATION_X
        ].default,
        prior_extrapolation_mode: Union[str, None] = v3ae_parameters[
            PRIOR_EXTRAPOLATION_MODE
        ].default,
        decoder_α_offset: float = v3ae_parameters[DECODER_α_OFFSET].default,
    ):
        super(V3AE, self).__init__(
            architecture,
            input_dims,
            activation,
            latent_dims=latent_dims,
            learning_rate=learning_rate,
            eps=eps,
            n_mc_samples=n_mc_samples,
        )

        self.β_elbo = 1

        self.τ_ood = τ_ood
        self.ood_z_generation_method = check_ood_z_generation_method(
            ood_z_generation_method
        )

        self.kde_bandwidth_multiplier = kde_bandwidth_multiplier

        self._switch_to_decoder_var = False
        self._student_t_decoder = True
        self._bernouilli_decoder = False

        self.decoder_α_offset = decoder_α_offset
        if self.architecture == FULLY_CONNECTED:
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
                ShiftLayer(1.0 + self.decoder_α_offset),
            )
            self.decoder_β = nn.Sequential(
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
            self.decoder_α = nn.Sequential(
                *base_decoder,
                decoder_conv_final(CONV_HIDDEN_DIMS[0], self.activation),
                nn.Softplus(),
                ShiftLayer(1.0 + self.decoder_α_offset),
            )
            self.decoder_β = nn.Sequential(
                *base_decoder,
                decoder_conv_final(CONV_HIDDEN_DIMS[0], self.activation),
                nn.Softplus(),
            )

        # set with `set_prior_parameters`
        self.prior_α = None
        self.prior_β = None

        self.prior_b = prior_b
        self.prior_epistemic_c = prior_epistemic_c
        self.prior_extrapolation_x = prior_extrapolation_x
        self.prior_extrapolation_mode = prior_extrapolation_mode
        if self.prior_extrapolation_x is not None:
            assert self.prior_extrapolation_mode in PRIOR_EXTRAPOLATION_AVAILABLE_MODES

        self._mean_x_train = None

        # PIG DL
        self.pig_dl = None

        # Inducing centroids for latent prior extrapolation
        self.N_inducing = 10
        self.inducing_centroids = torch.zeros((self.N_inducing, self.latent_size))

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
            "τ_ood",
            "ood_z_generation_method",
            "prior_b",
            "prior_epistemic_c",
            "prior_extrapolation_x",
            "prior_extrapolation_mode",
            "decoder_α_offset",
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
        x_train = []
        for idx, batch in enumerate(datamodule.train_dataloader()):
            x, _ = batch
            x_train.append(x)

        # Aggregate the whole dataset
        x_train = torch.cat(x_train, dim=0)
        x_train = torch.reshape(x_train, (-1, *datamodule.dims))

        # Save mean of dataset
        self._mean_x_train = x_train.mean(dim=0)[None, :]

        if (prior_α is None) & (prior_β is None):
            # %
            x_train_var = x_train.var(dim=0)

            # Per pixel prior mode
            prior_modes = 1 / x_train_var
            # Bound the modes
            prior_modes = torch.maximum(
                prior_modes, min_mode * torch.ones_like(prior_modes)
            )
            prior_modes = torch.minimum(
                prior_modes, max_mode * torch.ones_like(prior_modes)
            )
            inv_prior_modes = 1 / prior_modes

            if self.prior_epistemic_c is not None:
                prior_epistemic_c = self.prior_epistemic_c * torch.ones_like(
                    prior_modes
                )
                self.prior_α = prior_epistemic_c / (prior_epistemic_c - 1)
                self.prior_β = self.prior_α * inv_prior_modes / prior_epistemic_c
                self.prior_β = self.prior_β
            elif self.prior_b is not None:
                self.prior_β = self.prior_b * torch.ones_like(prior_modes).type_as(
                    x_train
                )
                self.prior_α = 1 + self.prior_β * prior_modes
            else:
                raise ValueError(
                    "No prior_α and prior_β provided, and no procedure for prior parameter selection provided."
                )

        elif (prior_α is not None) & (prior_β is not None):
            prior_α = float(prior_α) if isinstance(prior_α, int) else prior_α
            prior_β = float(prior_β) if isinstance(prior_β, int) else prior_β
            assert type(prior_α) == type(
                prior_β
            ), "prior_α and prior_β are not of the same type"
            if isinstance(prior_α, float) | isinstance(prior_α, int):
                self.prior_α = prior_α * torch.ones(datamodule.dims).type_as(x_train)
                self.prior_β = prior_β * torch.ones(datamodule.dims).type_as(x_train)
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
            raise ValueError("Incorrect prior values provided for prior_α and prior_β")

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
                # dm not registered on device
                x = x.to(self.device)
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
        agg_z = agg_z.to(self.device)
        agg_q_z_x_mean = agg_q_z_x_mean.to(self.device)
        agg_q_z_x_stddev = agg_q_z_x_stddev.to(self.device)
        # Make sure it's shuffled
        # Only works as such because shape is [len_train_dataset, latent_dims]
        perm_idx = torch.randperm(agg_z.shape[0])
        agg_z = agg_z[perm_idx]
        agg_q_z_x_mean = agg_q_z_x_mean[perm_idx]
        agg_q_z_x_stddev = agg_q_z_x_stddev[perm_idx]

        q_z_x = D.Independent(D.Normal(agg_q_z_x_mean, agg_q_z_x_stddev), 1)
        p_z_mean, p_z_stddev = (
            p_z_mean[0].repeat(agg_z.shape[0], 1),
            p_z_stddev[0].repeat(agg_z.shape[0], 1),
        )
        p_z_mean = p_z_mean.to(self.device)
        p_z_stddev = p_z_stddev.to(self.device)
        p_z = D.Independent(D.Normal(p_z_mean, p_z_stddev), 1)
        z_hat = self.generate_z_out(agg_z, q_z_x, p_z)
        # Create DL
        self.pig_dl = DataLoader(
            TensorDataset(z_hat), batch_size=self.dm.batch_size, shuffle=True
        )

        # Cleanup
        # Release vars
        del agg_z
        del agg_q_z_x_mean
        del agg_q_z_x_stddev
        del q_z_x
        del p_z_mean
        del p_z_stddev
        del p_z
        torch.cuda.empty_cache()
        # Safety, zero_grad on the optimisers
        for opt in self.optimizers():
            opt.zero_grad()
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
            # [len_train_dataset, latent_size]
            means, stddevs = (
                q_z_x.base_dist.mean.clone().detach().to(self.device),
                q_z_x.base_dist.stddev.clone().detach().to(self.device),
            )
            z_out = torch.zeros_like(means).type_as(means)
            # Need to batch the components - otherwise it's unfeasible to
            # Evaluate the aggregate posterior
            batch_size_components = self.dm.batch_size * 10
            n_batch_components = means.shape[0] // batch_size_components
            for i in range(n_batch_components + 1):
                _i = i * batch_size_components
                _i_1 = (i + 1) * batch_size_components
                _means, _stddevs = means[_i:_i_1], stddevs[_i:_i_1]
                agg_q_z_x = D.Independent(D.Normal(_means, _stddevs), 1)
                mix = D.Categorical(
                    torch.ones(
                        _means.shape[0],
                    ).to(self.device)
                )
                agg_q_z_x = D.MixtureSameFamily(mix, agg_q_z_x)
                q_start = D.Independent(
                    D.Normal(
                        q_z_x.mean[_i:_i_1],
                        2 * torch.ones_like(q_z_x.mean[_i:_i_1]),
                    ),
                    1,
                )
                z_start = q_start.sample((1,))[0]
                _z_out = density_gradient_descent(
                    agg_q_z_x,
                    z_start,
                    {"N_steps": gd_n_steps, "lr": gd_lr, "threshold": gd_threshold},
                )
                z_out[_i:_i_1] = _z_out.detach()
        else:
            z_out = torch.zeros_like(z)

        return z_out

    def _setup_inducing_centroids(self, N_inducing: int = 50):
        print("\n--- Generating Inducing Centroids", flush=True)
        # Gather all z
        with torch.no_grad():
            for idx, batch in enumerate(iter(self.dm.train_dataloader())):
                x, _ = batch
                # dm not registered on device
                x = x.to(self.device)
                # Actually don't need that, only keep the encoded z, with generation batch by batch
                _, _, _, _, _, z, _, _ = self._run_step(x)
                # Only keep one mc_sample
                z = z[0]
                if idx == 0:
                    agg_z = z
                else:
                    agg_z = torch.cat((agg_z, z), dim=0)
                agg_z = agg_z.to(self.device)
        # Kmean to get centroids
        kmeans = KMeans(n_clusters=N_inducing)
        kmeans.fit(agg_z.detach().cpu().numpy())
        self.inducing_centroids = torch.tensor(kmeans.cluster_centers_).type_as(agg_z)
        print("--- OK\n")

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
        λ, q_λ_z, p_λ = self.sample_precision(α_z, β_z, z)
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

    def sample_precision(self, alpha, beta, z):
        # batch_shape [n_mc_samples, BS] event_shape [input_size]
        # alpha = alpha + self.eps
        # beta = beta + self.eps
        q = D.Independent(D.Gamma(alpha, beta), 1)
        lbd = q.rsample()
        # Reshape prior to match [n_mc_samples, BS, input_size]
        prior_α = (
            self.prior_α.flatten()
            .repeat(alpha.shape[0], alpha.shape[1], 1)
            .type_as(alpha)
        )
        prior_β = (
            self.prior_β.flatten().repeat(beta.shape[0], beta.shape[1], 1).type_as(beta)
        )

        if (self.prior_extrapolation_mode == PRIOR_EXTRAPOLATION_ON_CENTROIDS) & (
            self.prior_extrapolation_x is not None
        ):
            prior_β_extrapolation = prior_β * self.prior_extrapolation_x
            s = self.get_latent_extrapolation_ratios(z)
            s = s.repeat(1, 1, beta.shape[2])
            # How to make sure that this works properly?
            # In analysis script, one could probably plot on the grid
            prior_β = ((1 - s) * prior_β) + (s * prior_β_extrapolation)

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
            # [n_mc_sample, BS, input_size]
            μ_z = p_x_z.mean
            α_z = p_x_z.base_dist.df / 2
            β_z = (p_x_z.base_dist.scale ** 2) * α_z

            expected_log_lambda = torch.digamma(α_z) - torch.log(β_z)
            expected_lambda = α_z / β_z
            # [n_mc_sample, BS]
            # sum over the independent input dims
            ellk_lbd = torch.sum(
                (1 / 2)
                * (expected_log_lambda - log_2_pi - expected_lambda * ((x - μ_z) ** 2)),
                dim=2,
            )

            # [BS]
            # MC integration over z samples
            ellk_lbd = torch.mean(ellk_lbd, dim=0)

            # [n_mc_sample, BS]
            kl_divergence_lbd = self.kl(q_λ_z, p_λ)
            # [BS]
            # MC integration over z samples
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

    def sample_z_out_like(self, z: torch.Tensor) -> torch.Tensor:
        # z might not fill the batch_size
        # [n_mc_samples, batch_size, latent_size]
        return next(iter(self.pig_dl))[0].type_as(z)[None, : z.shape[1], :]

    def ood_kl(
        self,
        p_λ: D.Distribution,
        z: torch.Tensor,
    ) -> torch.Tensor:

        if self.ood_z_generation_method in OOD_Z_GENERATION_AVAILABLE_METHODS:
            # [n_mc_samples, BS, *self.latent_dims]
            z_out = self.sample_z_out_like(z)
            z_out = z_out[: z.shape[1]]
            z_out = z_out.repeat(self.n_mc_samples, 1, 1)
            # [self.n_mc_samples, BS, self.input_size]
            _, _, α_z_out, β_z_out = self.parametrise_z(z_out)
            # batch_shape [self.n_mc_samples, BS] event_shape [self.input_size]
            _, q_λ_z_out, p_λ = self.sample_precision(α_z_out, β_z_out, z_out)
            # [self.n_mc_sample, BS]
            kl_divergence_lbd_out = self.kl(q_λ_z_out, p_λ)
            # [BS]
            kl_divergence_lbd_out = torch.mean(kl_divergence_lbd_out, dim=0)
            return kl_divergence_lbd_out
        return torch.zeros((1,))

    def update_hacks(self):
        if not self._refit_encoder_mode:
            previous_switch = copy.copy(self._switch_to_decoder_var)
            self._switch_to_decoder_var = (
                True if self.current_epoch >= self.trainer.max_epochs / 2 else False
            )
            self._student_t_decoder = self._switch_to_decoder_var
            self._bernouilli_decoder = not self._switch_to_decoder_var
            self.β_elbo = min(1, self.current_epoch / (self.trainer.max_epochs / 2))
            # Explicitely freeze the gradients of everything but alpha and beta
            if (
                self._switch_to_decoder_var
                and previous_switch != self._switch_to_decoder_var
            ):

                # ----
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

                self._setup_pi_dl()
                self._setup_inducing_centroids()

        # Handle refitting the encoder
        else:
            for m in self.decoder_μ.modules():
                if isinstance(m, nn.BatchNorm1d) | isinstance(m, nn.BatchNorm2d):
                    m.eval()
            for m in self.decoder_α.modules():
                if isinstance(m, nn.BatchNorm1d) | isinstance(m, nn.BatchNorm2d):
                    m.eval()
            for m in self.decoder_β.modules():
                if isinstance(m, nn.BatchNorm1d) | isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def step(self, batch, batch_idx, stage=None):
        x, _ = batch
        x_hat, p_x_z, λ, q_λ_z, p_λ, z, q_z_x, p_z = self._run_step(x)

        # [BS]
        expected_log_likelihood, ellk_lbd, kl_divergence_lbd = self.ellk(
            p_x_z, x, q_λ_z, p_λ
        )
        # [BS]
        kl_divergence_z = self.kl(q_z_x, p_z)

        # Expected value of the ELBO
        # Equivalently minimised with its sum.
        loss = -self.elbo(
            expected_log_likelihood, kl_divergence_z, train=(stage == TRAINING)
        ).mean()

        # Also verify that we are only training the decoder's variance
        kl_divergence_lbd_ood = -1 * torch.ones((1,))
        if (
            (stage == TRAINING)
            & (self.ood_z_generation_method is not None)
            & (self._student_t_decoder)
        ):
            p_λ_out = p_λ
            if (self.prior_extrapolation_mode == PRIOR_EXTRAPOLATION_ON_PI) & (
                self.prior_extrapolation_x is not None
            ):
                prior_β_extrapolation = p_λ.base_dist.rate * self.prior_extrapolation_x
                p_λ_out = D.Independent(
                    D.Gamma(
                        p_λ.base_dist.concentration,
                        prior_β_extrapolation,
                    ),
                    1,
                )
            # NOTE: beware, for understandability, tau is opposite.
            kl_divergence_lbd_ood = self.ood_kl(p_λ_out, z)
            expected_kl_divergence_lbd_ood = kl_divergence_lbd_ood.mean()
            loss = 2 * (
                (1 - self.τ_ood) * loss + self.τ_ood * expected_kl_divergence_lbd_ood
            )

        x_mean = batch_reshape(p_x_z.mean[0], self.input_dims)
        mean_rmse = torch.sqrt(F.mse_loss(x_mean, x))
        logs = {
            "llk": expected_log_likelihood.sum(),
            "ellk": expected_log_likelihood.mean(),
            "kl_z": kl_divergence_z.mean(),
            "ellk_lbd": ellk_lbd.mean(),
            "kl_lbd": kl_divergence_lbd.mean(),
            "kl_lbd_ood": kl_divergence_lbd_ood.mean(),
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
            # Samples OOD
            z_ood = self.sample_z_out_like(z)
            _, μ_z_ood, α_z_ood, β_z_ood = self.parametrise_z(z_ood)
            x_hat_ood, _ = self.sample_generative(μ_z_ood, α_z_ood, β_z_ood)
            x_hat_ood = batch_reshape(x_hat, self.input_dims)
            mean_x_train = self._mean_x_train.repeat(
                x_hat_ood.shape[0], 1, 1, 1
            ).type_as(x_hat_ood)
            logs[TEST_OOD_SAMPLE_FIT_MAE] = F.l1_loss(x_hat_ood, mean_x_train)
            logs[TEST_OOD_SAMPLE_FIT_RMSE] = torch.sqrt(
                F.mse_loss(x_hat_ood, mean_x_train)
            )
            # MSE Samples and their own mean OOD
            x_mean_ood = batch_reshape(μ_z_ood[0], self.input_dims)
            logs[TEST_OOD_SAMPLE_MEAN_MSE] = F.mse_loss(x_hat_ood, x_mean_ood)

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
            chain(self.decoder_α.parameters(), self.decoder_β.parameters()),
            lr=self.learning_rate,
        )
        return [model_opt, decoder_var_opt], []

    def get_latent_extrapolation_ratios(self, z: torch.Tensor) -> torch.Tensor:
        # [n_mc_samples, batch_size, latent_size]
        if self.device != self.inducing_centroids.device:
            self.inducing_centroids = self.inducing_centroids.to(self.device)
        dist, _ = torch.cdist(z, self.inducing_centroids).min(dim=2)
        s = locscale_sigmoid(dist[:, :, None], 6.907 * 0.3, 0.3)
        # [n_mc_samples, batch_size, 1]
        return s

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
        for p in self.decoder_α.parameters():
            p.requires_grad = False
        for m in self.decoder_α.modules():
            if isinstance(m, nn.BatchNorm1d) | isinstance(m, nn.BatchNorm2d):
                m.eval()
        for p in self.decoder_β.parameters():
            p.requires_grad = False
        for m in self.decoder_β.modules():
            if isinstance(m, nn.BatchNorm1d) | isinstance(m, nn.BatchNorm2d):
                m.eval()

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        return model_specific_args(v3ae_parameters, parent_parser)
