import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributions as tcd

from argparse import ArgumentParser

from sggm.definitions import regressor_parameters
from sggm.definitions import (
    β_ELBO,
    β_OOD,
    GAUSSIAN_NOISE_AROUND_X,
    EPS,
    OOD_X_GENERATION_METHOD,
    OPTIMISED_X_OOD,
    N_MC_SAMPLES,
)

# ----------
# Model definitions
# ----------
pi = torch.tensor([np.pi])
MARGINAL = "marginal"
POSTERIOR = "posterior"
available_methods = [MARGINAL, POSTERIOR]


def check_available_methods(method):
    assert (
        method in available_methods
    ), f"Unvalid method {method}, choices {available_methods}"


def fit_prior():
    # Heuristic for now
    prior_α = 1.02
    prior_β = 0.57
    return prior_α, prior_β


def BaseMLP(input_dim, hidden_dim):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.Sigmoid(),
        torch.nn.Linear(hidden_dim, 1),
    )


def BaseMLPSoftPlus(input_dim, hidden_dim):
    mod = BaseMLP(input_dim, hidden_dim)
    mod.add_module("softplus", torch.nn.Softplus())
    return mod


class ShiftLayer(torch.nn.Module):
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
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        prior_α: float,
        prior_β: float,
        β_elbo: float = regressor_parameters[β_ELBO].default,
        β_ood: float = regressor_parameters[β_OOD].default,
        ood_x_generation_method: str = regressor_parameters[
            OOD_X_GENERATION_METHOD
        ].default,
        eps: float = regressor_parameters[EPS].default,
        n_mc_samples: int = regressor_parameters[N_MC_SAMPLES].default,
    ):
        super(Regressor, self).__init__()

        # ---------
        # Parameters
        # ---------
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.eps = eps
        self.n_mc_samples = n_mc_samples

        self.ood_x_generation_method = ood_x_generation_method

        v_ini = torch.nn.Parameter(torch.Tensor([10 / 500]), requires_grad=True)
        self.register_parameter("v", v_ini)

        # ---------
        # HParameters
        # ---------

        self.prior_α = prior_α
        self.prior_β = prior_β

        self.β_ood = β_ood
        self.β_elbo = β_elbo

        self.lr = 1e-2
        self.lr_v = 1

        # ---------
        # Inference Networks
        # ---------
        self.μ = BaseMLP(input_dim, hidden_dim)

        self.α = BaseMLPSoftPlus(input_dim, hidden_dim)
        self.α.add_module("shift", ShiftLayer(1))

        self.β = BaseMLPSoftPlus(input_dim, hidden_dim)

        # ---------
        # Misc
        # ---------
        self.pp = tcd.Gamma(prior_α, prior_β)

        # Save hparams
        self.save_hyperparameters()

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
        var = torch.where(α > 1, self.β(x) / (α - 1), 1e20 * torch.ones(α.shape))
        return torch.sqrt(var)

    def predictive_mean(self, x: torch.Tensor, method: str = MARGINAL) -> torch.Tensor:
        check_available_methods(method)
        if method == MARGINAL:
            return self.marginal_predictive_mean(x)
        elif method == POSTERIOR:
            return self.posterior_predictive_mean(x)

    def predictive_std(self, x: torch.Tensor, method: str = MARGINAL) -> torch.Tensor:
        check_available_methods(method)
        if method == MARGINAL:
            return self.marginal_predictive_std(x)
        elif method == POSTERIOR:
            return self.posterior_predictive_std(x)

    def ood_x(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.ood_x_generation_method == GAUSSIAN_NOISE_AROUND_X:
            kl_noise_std = 20
            return x + kl_noise_std * torch.randn_like(x)
        elif self.ood_x_generation_method == OPTIMISED_X_OOD:
            kl = torch.mean(kwargs["kl"])
            kl_grad = torch.autograd.grad(kl, x, retain_graph=True)[0]
            random_direction = (torch.randint_like(kl_grad, 0, 2) * 2) - 1
            return x + self.v * random_direction * torch.sign(kl_grad)
        return torch.empty(0, 0)

    @staticmethod
    def llk(
        μ: torch.Tensor, α: torch.Tensor, β: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        expected_log_lambda = torch.digamma(α) - torch.log(β)
        expected_lambda = α / β
        ll = (1 / 2) * (
            expected_log_lambda - torch.log(2 * pi) - expected_lambda * ((y - μ) ** 2)
        )
        return ll

    @staticmethod
    def kl(
        α: torch.Tensor, β: torch.Tensor, a: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        qp = tcd.Gamma(α, β)
        pp = tcd.Gamma(a, b)
        return tcd.kl_divergence(qp, pp)

    def elbo(self, llk: torch.Tensor, kl: torch.Tensor) -> torch.Tensor:
        return torch.mean(llk - self.β_elbo * kl)

    # ---------
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [
                {"params": self.μ.parameters()},
                {"params": self.α.parameters()},
                {"params": self.β.parameters()},
                {"params": self.v, "lr": self.lr_v},
            ],
            lr=self.lr,
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        x.requires_grad = True
        μ_x, α_x, β_x = self(x)
        log_likelihood = self.llk(μ_x, α_x, β_x, y)
        kl_divergence = self.kl(α_x, β_x, self.prior_α, self.prior_β)
        x_out = self.ood_x(x, kl=kl_divergence)
        if torch.numel(x_out) > 0:
            _, α_x_out, β_x_out = self(x_out)
            kl_divergence_out = self.kl(α_x_out, β_x_out, self.prior_α, self.prior_β)
        else:
            kl_divergence_out = torch.zeros((1,))
        loss = -self.elbo(log_likelihood, kl_divergence) + self.β_ood * torch.mean(
            kl_divergence_out
        )
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        μ_x, α_x, β_x = self(x)
        log_likelihood = self.llk(μ_x, α_x, β_x, y)
        kl_divergence = self.kl(α_x, β_x, self.prior_α, self.prior_β)
        loss = -self.elbo(log_likelihood, kl_divergence)
        self.log("eval_loss", loss, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        μ_x, α_x, β_x = self(x)
        log_likelihood = self.llk(μ_x, α_x, β_x, y)
        kl_divergence = self.kl(α_x, β_x, self.prior_α, self.prior_β)
        loss = -self.elbo(log_likelihood, kl_divergence)
        self.log("test_loss", loss, on_step=True)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(
            parents=[parent_parser], add_help=False, conflict_handler="resolve"
        )
        for parameter in regressor_parameters.values():
            parser.add_argument(
                f"--{parameter.name}", default=parameter.default, type=parameter.type_
            )
        return parser
