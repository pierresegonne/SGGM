import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

from argparse import ArgumentParser

from sggm.definitions import (
    UNIFORM,
    model_specific_args,
    regressor_parameters,
    variational_regressor_parameters,
)
from sggm.definitions import (
    β_ELBO,
    τ_OOD,
    N_MC_SAMPLES,
    PRIOR_α,
    PRIOR_β,
    EPS,
    LEARNING_RATE,
    #
    SPLIT_TRAINING_MODE,
    PI_BATCH_SIZE_MULTIPLIER,
    #
    OOD_X_GENERATION_METHOD,
    BRUTE_FORCE,
    GAUSSIAN_NOISE,
    KDE,
    MEAN_SHIFT,
    #
    KDE_GD_LR,
    KDE_GD_N_STEPS,
    KDE_GD_THRESHOLD,
    MS_BW_FACTOR,
    MS_KDE_BW_FACTOR,
)
from sggm.definitions import (
    TRAIN_LOSS,
    EVAL_LOSS,
    TEST_LOSS,
    TEST_ELBO,
    TEST_MLLK,
    TEST_MEAN_FIT_MAE,
    TEST_MEAN_FIT_RMSE,
    TEST_VARIANCE_FIT_MAE,
    TEST_VARIANCE_FIT_RMSE,
    TEST_SAMPLE_FIT_MAE,
    TEST_SAMPLE_FIT_RMSE,
    TEST_ELLK,
    TEST_KL,
    NOISE_UNCERTAINTY,
    NOISE_KL,
)
from sggm.model_helper import log_2_pi, get_activation_function, ShiftLayer
from sggm.regression_model_helper import (
    check_mixture_ratio,
    check_ood_x_generation_method,
    generate_noise_for_model_test,
    gaussian_noise_pig_dl,
    kde_pig_dl,
    mean_shift_pig_dl,
)

# ----------
# Model definitions
# ----------

MARGINAL = "marginal"
POSTERIOR = "posterior"
available_methods = [MARGINAL, POSTERIOR]


def check_available_methods(method: str):
    assert (
        method in available_methods
    ), f"Unvalid method {method}, choices {available_methods}"


def BaseMLP(
    input_dim: int, hidden_dim: int, out_dim: int, activation: str
) -> nn.Module:
    f = get_activation_function(activation)
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        f,
        nn.Linear(hidden_dim, out_dim),
    )


def BaseMLPSoftPlus(
    input_dim: int, hidden_dim: int, out_dim: int, activation: str
) -> nn.Module:
    mod = BaseMLP(input_dim, hidden_dim, out_dim, activation)
    mod.add_module("softplus", nn.Softplus())
    return mod


# ----------
# Model
# ----------
# Note that the default values are provided to ease exploration, they are actually not used.


class Regressor(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        return model_specific_args(regressor_parameters, parent_parser)


class VariationalRegressor(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        out_dim: int,
        activation: str,
        learning_rate: float = variational_regressor_parameters[LEARNING_RATE].default,
        prior_α: float = variational_regressor_parameters[PRIOR_α].default,
        prior_β: float = variational_regressor_parameters[PRIOR_β].default,
        β_elbo: float = variational_regressor_parameters[β_ELBO].default,
        τ_ood: float = variational_regressor_parameters[τ_OOD].default,
        ood_x_generation_method: str = variational_regressor_parameters[
            OOD_X_GENERATION_METHOD
        ].default,
        eps: float = variational_regressor_parameters[EPS].default,
        n_mc_samples: int = variational_regressor_parameters[N_MC_SAMPLES].default,
        y_mean: float = 0.0,  # Not in regressor parameters as it is infered from data
        y_std: float = 1.0,  # Not in regressor parameters as it is infered from data
        split_training_mode: str = variational_regressor_parameters[
            SPLIT_TRAINING_MODE
        ].default,
        pi_batch_size_multiplier: float = variational_regressor_parameters[
            PI_BATCH_SIZE_MULTIPLIER
        ].default,
        ms_bw_factor: float = variational_regressor_parameters[MS_BW_FACTOR].default,
        ms_kde_bw_factor: float = variational_regressor_parameters[
            MS_KDE_BW_FACTOR
        ].default,
        kde_gd_n_steps: float = variational_regressor_parameters[
            KDE_GD_N_STEPS
        ].default,
        kde_gd_lr: float = variational_regressor_parameters[KDE_GD_LR].default,
        kde_gd_threshold: float = variational_regressor_parameters[
            KDE_GD_THRESHOLD
        ].default,
    ):
        super(VariationalRegressor, self).__init__()

        # ---------
        # Parameters
        # ---------
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.activation = activation

        self.eps = eps
        self.n_mc_samples = n_mc_samples

        # OOD
        self.ood_x_generation_method = check_ood_x_generation_method(
            ood_x_generation_method
        )
        self.pi_batch_size_multiplier = pi_batch_size_multiplier
        # Mean shift
        self.ms_bw_factor = ms_bw_factor
        self.ms_kde_bw_factor = ms_kde_bw_factor
        # KDE gd pig
        self.kde_gd_n_steps = kde_gd_n_steps
        self.kde_gd_lr = kde_gd_lr
        self.kde_gd_threshold = kde_gd_threshold

        # ---------
        # HParameters
        # ---------
        self.prior_α = prior_α
        self.prior_β = prior_β

        self.β_elbo = β_elbo
        self.τ_ood = check_mixture_ratio(τ_ood)

        self.learning_rate = learning_rate

        # ---------
        # Inference Networks
        # ---------
        self.μ = BaseMLP(input_dim, hidden_dim, out_dim, activation)

        self.α = BaseMLPSoftPlus(input_dim, hidden_dim, out_dim, activation)
        self.α.add_module("shift", ShiftLayer(1))

        self.β = BaseMLPSoftPlus(input_dim, hidden_dim, out_dim, activation)

        # ---------
        # Misc
        # ---------
        self.example_input_array = torch.rand((10, self.input_dim))

        self.split_training_mode = split_training_mode
        self.mse_mode = False

        # Save hparams
        self.save_hyperparameters(
            "input_dim",
            "hidden_dim",
            "out_dim",
            "learning_rate",
            "activation",
            "prior_α",
            "prior_β",
            "β_elbo",
            "τ_ood",
            "ood_x_generation_method",
            "eps",
            "n_mc_samples",
            "split_training_mode",
            "kde_gd_n_steps",
            "kde_gd_lr",
            "kde_gd_threshold",
            "pi_batch_size_multiplier",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.μ(x), self.α(x), self.β(x)

    def posterior_predictive_mean(self, x: torch.Tensor) -> torch.Tensor:
        return self.μ(x)

    def posterior_predictive_std(
        self, x: torch.Tensor, exact: bool = True
    ) -> torch.Tensor:
        if exact:
            mean_precision = self.α(x) / self.β(x)
            σ = 1 / torch.sqrt(mean_precision + self.eps)
        else:
            qp = D.Gamma(self.α(x), self.β(x))
            samples_precision = qp.rsample(torch.Size([self.n_mc_samples]))
            precision = torch.mean(samples_precision, 0, True)
            σ = 1 / torch.sqrt(precision)
        return σ

    def prior_std(self, x: torch.Tensor) -> torch.Tensor:
        α = self.prior_α * torch.ones((x.shape[0], 1))
        β = self.prior_β * torch.ones((x.shape[0], 1))
        return torch.sqrt(β / (α - 1))

    def marginal_predictive_mean(self, x: torch.Tensor) -> torch.Tensor:
        return self.μ(x)

    def marginal_predictive_std(self, x: torch.Tensor) -> torch.Tensor:
        α = self.α(x)
        var = torch.where(
            α > 1, self.β(x) / (α - 1), 1e20 * torch.ones(α.shape).type_as(x)
        )
        return torch.sqrt(var)

    def predictive_mean(
        self, x: torch.Tensor, method: str = MARGINAL, scaled: bool = False
    ) -> torch.Tensor:
        check_available_methods(method)
        if method == MARGINAL:
            pred_mean = self.marginal_predictive_mean(x)
        elif method == POSTERIOR:
            pred_mean = self.posterior_predictive_mean(x)

        if scaled:
            pred_mean = self.y_mean + pred_mean * self.y_std

        return pred_mean

    def predictive_std(
        self, x: torch.Tensor, method: str = MARGINAL, scaled: bool = False
    ) -> torch.Tensor:
        check_available_methods(method)
        if method == MARGINAL:
            pred_std = self.marginal_predictive_std(x)
        elif method == POSTERIOR:
            pred_std = self.posterior_predictive_std(x)

        if scaled:
            pred_std *= self.y_std

        return pred_std

    def setup_pig(self, dm: pl.LightningDataModule) -> None:
        """ NOTE: Memory bottleneck here as following methods load entire dataset on memory. """
        N_hat_multiplier = 1
        pi_batch_size = int(dm.batch_size * self.pi_batch_size_multiplier)

        if self.ood_x_generation_method == GAUSSIAN_NOISE:
            self.pig_dl = gaussian_noise_pig_dl(
                dm,
                pi_batch_size,
                N_hat_multiplier=N_hat_multiplier,
                sigma_multiplier=3,
            )

        elif self.ood_x_generation_method == KDE:
            self.pig_dl = kde_pig_dl(
                dm,
                pi_batch_size,
                N_hat_multiplier=N_hat_multiplier,
                gd_lr=self.kde_gd_lr,
                gd_n_steps=self.kde_gd_n_steps,
                gd_threshold=self.kde_gd_threshold,
            )

        elif self.ood_x_generation_method == MEAN_SHIFT:
            # Assigns a pig datamodule
            self.pig_dl = mean_shift_pig_dl(
                dm,
                pi_batch_size,
                N_hat_multiplier=N_hat_multiplier,
                max_iters=100,
                # ad hoc factors
                h_factor=self.ms_bw_factor,
                sigma_factor=self.ms_kde_bw_factor,
            )

    def ood_x(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # Skip generation if we're not going to see it used
        if self.τ_ood > 0:
            # Special case - can only be done "online"
            if self.ood_x_generation_method == BRUTE_FORCE:
                x_ood_proposal = torch.reshape(
                    torch.linspace(-25, 35, 4000), (4000, 1)
                ).type_as(x)
                _, alpha_ood, τ_ood = self(x_ood_proposal)
                kl = self.kl(alpha_ood, τ_ood, self.prior_α, self.prior_β).detach()

                # objective = kl - llk
                objective = kl

                # Top K and then subsample
                _, idx = torch.topk(objective, 1000, dim=0, sorted=False)
                x_ood = x_ood_proposal[idx][::2]
                return torch.reshape(x_ood, (500, 1))

            elif self.ood_x_generation_method == UNIFORM:
                N = x.shape[0]
                x_right = torch.FloatTensor(int(N / 2), 1).uniform_(-200, -190)
                x_left = torch.FloatTensor(int(N / 2), 1).uniform_(190, 200)
                return torch.cat((x_right, x_left), dim=0)

            # General case, PIG DL available.
            # For now, hack to make it work without working for analysis
            # Ok get the same number of pi as training points
            elif getattr(self, "pig_dl", None):
                x_ood = next(iter(self.pig_dl))[0]
                return x_ood.type_as(x)

        return torch.empty(0, 0)

    def tune_on_validation(self, x: torch.Tensor, **kwargs):
        pass

    @staticmethod
    def ellk(
        μ: torch.Tensor,
        α: torch.Tensor,
        β: torch.Tensor,
        y: torch.Tensor,
        ε: float = 1e-10,
    ) -> torch.Tensor:
        β = β + ε
        expected_log_lambda = torch.digamma(α) - torch.log(β)
        expected_lambda = α / β
        ll = (1 / 2) * (
            expected_log_lambda - log_2_pi - expected_lambda * ((y - μ) ** 2)
        )
        return ll

    @staticmethod
    def kl(
        α: torch.Tensor,
        β: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        ε: float = 1e-10,
    ) -> torch.Tensor:
        β = β + ε
        qp = D.Gamma(α, β)
        pp = D.Gamma(a, b)
        return D.kl_divergence(qp, pp)

    def elbo(
        self, ellk: torch.Tensor, kl: torch.Tensor, train: bool = True
    ) -> torch.Tensor:
        β = self.β_elbo if train else 1
        return torch.mean(ellk - β * kl)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x.requires_grad = True

        μ_x, α_x, β_x = self(x)

        # Split training only mse
        if self.mse_mode:
            # default reduction is mean
            loss = F.mse_loss(μ_x, y)
        else:
            expected_log_likelihood = self.ellk(μ_x, α_x, β_x, y)
            kl_divergence = self.kl(α_x, β_x, self.prior_α, self.prior_β)

            x_out = self.ood_x(x, kl=kl_divergence)
            x.requires_grad = False
            if torch.numel(x_out) > 0:
                μ_x_out, α_x_out, β_x_out = self(x_out)
                expected_log_likelihood_out = self.ellk(
                    μ_x_out, α_x_out, β_x_out, μ_x_out
                )
                kl_divergence_out = self.kl(
                    α_x_out, β_x_out, self.prior_α, self.prior_β
                )
            else:
                expected_log_likelihood_out = torch.zeros((1,)).type_as(x)
                kl_divergence_out = torch.zeros((1,)).type_as(x)

            # NOTE: careful, it's reverse definition for τ
            loss = -(1 - self.τ_ood) * self.elbo(
                expected_log_likelihood, kl_divergence
            ) + self.τ_ood * torch.mean(kl_divergence_out)

            # 1D data, recorded P.I histogram
            if (torch.numel(x_out) > 0) and (x_out.shape[1] == 1):
                self.logger.experiment.add_histogram("x_out", x_out, self.current_epoch)

        self.log(TRAIN_LOSS, loss, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        if self.mse_mode:
            μ_x, _, _ = self(x)
            loss = F.mse_loss(μ_x, y)
        else:
            with torch.set_grad_enabled(True):
                x.requires_grad = True
                μ_x, α_x, β_x = self(x)
                log_likelihood = self.ellk(μ_x, α_x, β_x, y)
                kl_divergence = self.kl(α_x, β_x, self.prior_α, self.prior_β)
                self.tune_on_validation(x, kl=kl_divergence)
            # --
            loss = -self.elbo(log_likelihood, kl_divergence, train=False)

        self.log(EVAL_LOSS, loss, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        μ_x, α_x, β_x = self(x)
        log_likelihood = self.ellk(μ_x, α_x, β_x, y)
        kl_divergence = self.kl(α_x, β_x, self.prior_α, self.prior_β)

        if self.mse_mode:
            loss = F.mse_loss(μ_x, y)
        else:
            loss = -self.elbo(log_likelihood, kl_divergence, train=False)

        y_pred = self.predictive_mean(x)

        m_p = D.StudentT(2 * α_x, loc=μ_x, scale=torch.sqrt(β_x / α_x))

        # ---------
        # Metrics
        self.log(TEST_LOSS, loss, on_epoch=True)
        self.log(TEST_ELBO, -loss, on_epoch=True)
        self.log(
            TEST_MLLK, torch.sum(m_p.log_prob(y)), on_epoch=True
        )  # i.i.d assumption

        # Mean fit
        self.log(TEST_MEAN_FIT_MAE, F.l1_loss(y_pred, y), on_epoch=True)
        self.log(TEST_MEAN_FIT_RMSE, torch.sqrt(F.mse_loss(y_pred, y)), on_epoch=True)

        # Variance fit
        pred_var = self.predictive_std(x) ** 2
        empirical_var = (y_pred - y) ** 2
        self.log(
            TEST_VARIANCE_FIT_MAE, F.l1_loss(pred_var, empirical_var), on_epoch=True
        )
        self.log(
            TEST_VARIANCE_FIT_RMSE,
            torch.sqrt(F.mse_loss(pred_var, empirical_var)),
            on_epoch=True,
        )

        # Sample fit
        ancestral = False
        if ancestral:
            lbds = D.Gamma(α_x, β_x).sample((1,))
            samples_y = (
                D.Normal(μ_x, 1 / torch.sqrt(lbds)).sample((1,)).reshape(y.shape)
            )
        else:
            samples_y = m_p.sample((1,)).reshape(y.shape)

        self.log(TEST_SAMPLE_FIT_MAE, F.l1_loss(samples_y, y), on_epoch=True)
        self.log(
            TEST_SAMPLE_FIT_RMSE, torch.sqrt(F.mse_loss(samples_y, y)), on_epoch=True
        )

        # Model expected log likelihood
        self.log(TEST_ELLK, torch.mean(log_likelihood), on_epoch=True)

        # Model KL
        self.log(TEST_KL, torch.mean(kl_divergence), on_epoch=True)

        # Noise
        x_noisy = generate_noise_for_model_test(x)
        μ_x, α_x, β_x = self(x_noisy)
        sigma = self.predictive_std(x_noisy)
        kl_divergence = self.kl(α_x, β_x, self.prior_α, self.prior_β)

        # Noise likelihood
        self.log(NOISE_UNCERTAINTY, torch.mean(sigma), on_epoch=True)

        # Noise KL
        self.log(NOISE_KL, torch.mean(kl_divergence), on_epoch=True)
        return loss

    # ---------
    def configure_optimizers(self):
        params = [
            {"params": self.μ.parameters()},
            {"params": self.α.parameters()},
            {"params": self.β.parameters()},
        ]
        optimizer = torch.optim.Adam(
            params,
            lr=self.learning_rate,
        )
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        return model_specific_args(variational_regressor_parameters, parent_parser)
