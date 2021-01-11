import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributions as tcd
import torch.nn as nn
import torch.nn.functional as F

from argparse import ArgumentParser

# TODO remove if likelihood penalty is inconclusive
from sklearn.mixture import GaussianMixture

from sggm.definitions import (
    model_specific_args,
    regressor_parameters,
    variational_regressor_parameters,
)
from sggm.definitions import (
    β_ELBO,
    β_OOD,
    N_MC_SAMPLES,
    PRIOR_α,
    PRIOR_β,
    EPS,
    LEARNING_RATE,
    #
    OOD_X_GENERATION_METHOD,
    ADVERSARIAL,
    BRUTE_FORCE,
    GAUSSIAN_NOISE,
    UNIFORM,
    V_PARAM,
)
from sggm.definitions import (
    ACTIVATION_FUNCTIONS,
    F_ELU,
    F_RELU,
    F_SIGMOID,
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
    NOISE_ELLK,
    NOISE_KL,
)
from sggm.regression_model_helper import generate_noise_for_model_test

# ----------
# Model definitions
# ----------
log_2_pi = float(torch.log(2 * torch.tensor([np.pi])))
MARGINAL = "marginal"
POSTERIOR = "posterior"
available_methods = [MARGINAL, POSTERIOR]


def check_available_methods(method):
    assert (
        method in available_methods
    ), f"Unvalid method {method}, choices {available_methods}"


def BaseMLP(input_dim, hidden_dim, activation):
    assert (
        activation in ACTIVATION_FUNCTIONS
    ), f"activation_function={activation} is not in {ACTIVATION_FUNCTIONS}"
    if activation == F_ELU:
        f = nn.ELU()
    elif activation == F_RELU:
        f = nn.ReLU()
    elif activation == F_SIGMOID:
        f = nn.Sigmoid()
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        f,
        nn.Linear(hidden_dim, 1),
    )


def BaseMLPSoftPlus(input_dim, hidden_dim, activation):
    mod = BaseMLP(input_dim, hidden_dim, activation)
    mod.add_module("softplus", nn.Softplus())
    return mod


class ShiftLayer(nn.Module):
    def __init__(self, shift_factor):
        super(ShiftLayer, self).__init__()
        self.shift_factor = shift_factor

    def forward(self, x):
        return self.shift_factor + x


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
        activation: str,
        learning_rate: float = variational_regressor_parameters[LEARNING_RATE].default,
        prior_α: float = variational_regressor_parameters[PRIOR_α].default,
        prior_β: float = variational_regressor_parameters[PRIOR_β].default,
        β_elbo: float = variational_regressor_parameters[β_ELBO].default,
        β_ood: float = variational_regressor_parameters[β_OOD].default,
        ood_x_generation_method: str = variational_regressor_parameters[
            OOD_X_GENERATION_METHOD
        ].default,
        eps: float = variational_regressor_parameters[EPS].default,
        n_mc_samples: int = variational_regressor_parameters[N_MC_SAMPLES].default,
        y_mean: float = 0.0,  # Not in regressor parameters as it is infered from data
        y_std: float = 1.0,  # Not in regressor parameters as it is infered from data
    ):
        super(VariationalRegressor, self).__init__()

        # ---------
        # Parameters
        # ---------
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.activation = activation

        self.eps = eps
        self.n_mc_samples = n_mc_samples

        # OOD
        self.ood_x_generation_method = ood_x_generation_method
        if self.ood_x_generation_method == ADVERSARIAL:
            self.ood_generator_v = 1
        elif self.ood_x_generation_method == V_PARAM:
            v_ini = nn.Parameter(
                0.1 * torch.ones((1, self.input_dim)), requires_grad=True
            )
            self.register_parameter("ood_generator_v", v_ini)
        else:
            self.ood_generator_v = None

        # ---------
        # HParameters
        # ---------
        self.prior_α = prior_α
        self.prior_β = prior_β

        self.β_ood = β_ood
        self.β_elbo = β_elbo

        self.learning_rate = learning_rate
        self.lr_v = 1e-2
        self.lr_ga_kl = 1

        # ---------
        # Inference Networks
        # ---------
        self.μ = BaseMLP(input_dim, hidden_dim, activation)

        self.α = BaseMLPSoftPlus(input_dim, hidden_dim, activation)
        self.α.add_module("shift", ShiftLayer(1))

        self.β = BaseMLPSoftPlus(input_dim, hidden_dim, activation)

        # ---------
        # Misc
        # ---------
        self.pp = tcd.Gamma(prior_α, prior_β)
        self.example_input_array = torch.rand((10, self.input_dim))

        self._marginal_loss = False
        self._ood_ellk = False

        # Save hparams
        self.save_hyperparameters(
            "input_dim",
            "hidden_dim",
            "learning_rate",
            "activation",
            "prior_α",
            "prior_β",
            "β_elbo",
            "β_ood",
            "ood_x_generation_method",
            "eps",
            "n_mc_samples",
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
            qp = tcd.Gamma(self.α(x), self.β(x))
            samples_precision = qp.rsample(torch.Size([self.n_mc_samples]))
            precision = torch.mean(samples_precision, 0, True)
            σ = 1 / torch.sqrt(precision)
        return σ

    def marginal_predictive_mean(self, x: torch.Tensor) -> torch.Tensor:
        return self.μ(x)

    def marginal_predictive_std(self, x: torch.Tensor) -> torch.Tensor:
        α = self.α(x)
        # np.inf is not a number
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

    def ood_x(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.ood_x_generation_method == GAUSSIAN_NOISE:
            noise_std = torch.std(x) * 3  # 3 is Arbitrary
            return x + noise_std * torch.randn_like(x)

        elif self.ood_x_generation_method == V_PARAM:
            kl = torch.mean(kwargs["kl"])
            kl_grad = torch.autograd.grad(kl, x, retain_graph=True)[0]
            normed_kl_grad = kl_grad / torch.linalg.norm(kl_grad, dim=1)[:, None]
            return x + self.ood_generator_v * normed_kl_grad

        elif self.ood_x_generation_method == ADVERSARIAL:
            kl = torch.mean(kwargs["kl"])
            kl_grad = torch.autograd.grad(kl, x, retain_graph=True)[0]
            # By default norm 2 or fro
            normed_kl_grad = kl_grad / torch.linalg.norm(kl_grad, dim=1)[:, None]
            return x + self.ood_generator_v * normed_kl_grad

        elif self.ood_x_generation_method == UNIFORM:
            raise NotImplementedError(
                "Uniform X ood generation must be implemented per use case."
            )

        elif self.ood_x_generation_method == BRUTE_FORCE:
            x_ood_proposal = torch.reshape(
                torch.linspace(-25, 35, 4000), (4000, 1)
            ).type_as(x)
            _, alpha_ood, beta_ood = self(x_ood_proposal)
            kl = self.kl(alpha_ood, beta_ood, self.prior_α, self.prior_β).detach()

            # density likelihood penalty
            # TODO remove once investigation completed
            # gm = GaussianMixture(n_components=5).fit(x.detach().cpu())
            # llk = gm.score_samples(x_ood_proposal.detach().cpu()).reshape(-1, 1)
            # llk = torch.Tensor(llk).to(self.device)

            # objective = kl - llk
            objective = kl

            # Top K and then subsample
            _, idx = torch.topk(objective, 1000, dim=0, sorted=False)
            x_ood = x_ood_proposal[idx][::2]
            return torch.reshape(x_ood, (500, 1))

        return torch.empty(0, 0)

    def tune_on_validation(self, x: torch.Tensor, **kwargs):
        if self.ood_x_generation_method == ADVERSARIAL:
            kl = torch.mean(kwargs["kl"])
            kl_grad = torch.autograd.grad(kl, x, retain_graph=True)[0]
            normed_kl_grad = kl_grad / torch.linalg.norm(kl_grad, dim=1)[:, None]

            with torch.no_grad():
                v_available = [0.0001, 0.001, 0.01, 0.1, 1, 2, 3, 5, 10, 25, 100]
                v_, kl_ = None, -np.inf
                for v_proposal in v_available:

                    _, α, β = self(x + v_proposal * normed_kl_grad)
                    kl_proposal = torch.mean(self.kl(α, β, self.prior_α, self.prior_β))
                    if (kl_proposal > kl_) & (kl_proposal < 1e10):
                        kl_ = kl_proposal
                        v_ = v_proposal

                # Prevent divergence
                if (kl_ > -np.inf) & (v_ is not None):
                    self.ood_generator_v = v_

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
        qp = tcd.Gamma(α, β)
        pp = tcd.Gamma(a, b)
        return tcd.kl_divergence(qp, pp)

    def elbo(
        self, ellk: torch.Tensor, kl: torch.Tensor, train: bool = True
    ) -> torch.Tensor:
        β = self.β_elbo if train else 1
        return torch.mean(ellk - β * kl)

    def training_step(self, batch, batch_idx):

        x, y = batch
        x.requires_grad = True

        μ_x, α_x, β_x = self(x)

        expected_log_likelihood = self.ellk(μ_x, α_x, β_x, y)
        kl_divergence = self.kl(α_x, β_x, self.prior_α, self.prior_β)

        x_out = self.ood_x(x, kl=kl_divergence)
        x.requires_grad = False
        if torch.numel(x_out) > 0:
            μ_x_out, α_x_out, β_x_out = self(x_out)
            expected_log_likelihood_out = self.ellk(μ_x_out, α_x_out, β_x_out, μ_x_out)
            kl_divergence_out = self.kl(α_x_out, β_x_out, self.prior_α, self.prior_β)
        else:
            expected_log_likelihood_out = torch.zeros((1,)).type_as(x)
            kl_divergence_out = torch.zeros((1,)).type_as(x)

        loss = -self.elbo(
            expected_log_likelihood, kl_divergence
        ) + self.β_ood * torch.mean(kl_divergence_out)

        if self._ood_ellk:
            nellk_out = -expected_log_likelihood_out
            loss = -self.elbo(
                expected_log_likelihood, kl_divergence
            ) - self.β_ood * self.elbo(nellk_out, kl_divergence_out)

        # Define the marginal y|x
        if self._marginal_loss:
            m_p = tcd.StudentT(2 * α_x, loc=μ_x, scale=torch.sqrt(β_x / α_x))
            loss = -torch.mean(m_p.log_prob(y)) + self.β_ood * torch.mean(
                kl_divergence_out
            )

        if (torch.numel(x_out) > 0) and (x_out.shape[1] == 1):
            self.logger.experiment.add_histogram("x_out", x_out, self.current_epoch)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        with torch.set_grad_enabled(True):
            x.requires_grad = True
            μ_x, α_x, β_x = self(x)
            log_likelihood = self.ellk(μ_x, α_x, β_x, y)
            kl_divergence = self.kl(α_x, β_x, self.prior_α, self.prior_β)
            # --
            # Avoid exploding gradients
            self.tune_on_validation(x, kl=kl_divergence)
        # --
        loss = -self.elbo(log_likelihood, kl_divergence, train=False)

        # Define the marginal y|x
        if self._marginal_loss:
            m_p = tcd.StudentT(2 * α_x, loc=μ_x, scale=torch.sqrt(β_x / α_x))
            loss = -torch.mean(m_p.log_prob(y))

        self.log(EVAL_LOSS, loss, on_epoch=True)
        if self.ood_generator_v is not None:
            self.log("ood_generator_v", self.ood_generator_v, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        μ_x, α_x, β_x = self(x)
        log_likelihood = self.ellk(μ_x, α_x, β_x, y)
        kl_divergence = self.kl(α_x, β_x, self.prior_α, self.prior_β)
        loss = -self.elbo(log_likelihood, kl_divergence, train=False)
        y_pred = self.predictive_mean(x)

        m_p = tcd.StudentT(2 * α_x, loc=μ_x, scale=torch.sqrt(β_x / α_x))
        # Define the marginal y|x
        if self._marginal_loss:
            loss = -torch.mean(m_p.log_prob(y))
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
            lbds = tcd.Gamma(α_x, β_x).sample((1,))
            samples_y = (
                tcd.Normal(μ_x, 1 / torch.sqrt(lbds)).sample((1,)).reshape(y.shape)
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
        log_likelihood = self.ellk(μ_x, α_x, β_x, μ_x)  # assume perfect match OOD
        kl_divergence = self.kl(α_x, β_x, self.prior_α, self.prior_β)

        # Noise likelihood
        self.log(NOISE_ELLK, torch.mean(log_likelihood), on_epoch=True)

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
        if self.ood_x_generation_method == V_PARAM:
            params += [
                {"params": self.ood_generator_v, "lr": self.lr_v},
            ]
        optimizer = torch.optim.Adam(
            params,
            lr=self.learning_rate,
        )
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        return model_specific_args(variational_regressor_parameters, parent_parser)
