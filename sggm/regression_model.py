import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributions as tcd

from argparse import ArgumentParser

from sggm.definitions import regressor_parameters
from sggm.definitions import β_OUT, EPS, N_MC_SAMPLES

pi = torch.tensor([np.pi])


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


class Regressor(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        prior_α,
        prior_β,
        β_out=regressor_parameters[β_OUT],
        eps=regressor_parameters[EPS],
        n_mc_samples=regressor_parameters[N_MC_SAMPLES],
    ):
        super(Regressor, self).__init__()

        # ---------
        # Parameters
        # ---------
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.prior_α = prior_α
        self.prior_β = prior_β
        self.pp = tcd.Gamma(prior_α, prior_β)

        self.β_out = β_out

        self.lr = 1e-2

        self.eps = eps
        self.n_mc_samples = n_mc_samples

        v_ini = torch.nn.Parameter(torch.Tensor([10 / 500]), requires_grad=True)
        self.register_parameter("v", v_ini)
        self.lr_v = 1

        # ---------
        # Inference Networks
        # ---------
        self.μ = BaseMLP(input_dim, hidden_dim)

        self.α = BaseMLPSoftPlus(input_dim, hidden_dim)
        self.α.add_module("shift", ShiftLayer(1))

        self.β = BaseMLPSoftPlus(input_dim, hidden_dim)

    def forward(self, x):
        return self.μ(x), self.α(x), self.β(x)

    def posterior_predictive_mean(self, x):
        return self.μ(x)

    def posterior_predictive_std(self, x, exact=True):
        if exact:
            mean_precision = self.α(x) / self.β(x)
            σ = 1 / torch.sqrt(mean_precision + self.eps)
        else:
            qp = tcd.Gamma(self.α(x), self.β(x))
            samples_precision = qp.rsample(torch.Size([self.n_mc_samples]))
            precision = torch.mean(samples_precision, 0, True)
            σ = 1 / torch.sqrt(precision)
        return σ

    def marginal_predictive_mean(self, x):
        return self.μ(x)

    def marginal_predictive_std(self, x):
        α = self.α(x)
        # np.inf is not a number
        var = torch.where(α > 1, self.β(x) / (α - 1), 1e20 * torch.ones(α.shape))
        return torch.sqrt(var)

    def ood_x(self, x, **kwargs):
        kl = kwargs["kl"]
        kl_grad = torch.autograd.grad(kl, x, retain_graph=True)[0]
        random_direction = (torch.randint_like(kl_grad, 0, 2) * 2) - 1
        return x + self.v * random_direction * torch.sign(kl_grad)

    @staticmethod
    def llk(μ, α, β, y):
        expected_log_lambda = torch.digamma(α) - torch.log(β)
        expected_lambda = α / β
        ll = (1 / 2) * (
            expected_log_lambda - torch.log(2 * pi) - expected_lambda * ((y - μ) ** 2)
        )
        return ll

    @staticmethod
    def kl(α, β, a, b):
        qp = tcd.Gamma(α, β)
        pp = tcd.Gamma(a, b)
        return tcd.kl_divergence(qp, pp)

    @staticmethod
    def elbo(llk, kl):
        return torch.mean(llk - kl)

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
        _, α_x_out, β_x_out = self(x_out)
        kl_divergence_out = self.kl(α_x_out, β_x_out, self.prior_α, self.prior_β)
        loss = self.elbo(log_likelihood, kl_divergence) + self.β_out * torch.mean(
            kl_divergence_out
        )
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x.requires_grad = True
        μ_x, α_x, β_x = self(x)
        log_likelihood = self.llk(μ_x, α_x, β_x, y)
        kl_divergence = self.kl(α_x, β_x, self.prior_α, self.prior_β)
        loss = self.elbo(log_likelihood, kl_divergence)
        self.log("eval_loss", loss, on_step=True)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        for parameter in regressor_parameters.values():
            parser.add_argument(
                f"--{parameter.name}", default=parameter.default, type=parameter.type
            )
        return parser
