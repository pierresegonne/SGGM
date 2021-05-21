import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

from argparse import ArgumentParser
from copy import copy
from itertools import chain
from typing import List, Tuple, Union

from sggm.definitions import (
    EVAL_LOSS,
    NOISE_UNCERTAINTY,
    N_ENS,
    TEST_LOSS,
    TEST_ELLK,
    TEST_MEAN_FIT_MAE,
    TEST_MEAN_FIT_RMSE,
    TEST_VARIANCE_FIT_MAE,
    TEST_VARIANCE_FIT_RMSE,
    TRAIN_LOSS,
)
from sggm.definitions import DROPOUT_RATE, EPS, LEARNING_RATE, N_MC_SAMPLES
from sggm.definitions import (
    ens_regressor_parameters,
    mcd_regressor_parameters,
    model_specific_args,
)
from sggm.model_helper import get_activation_function
from sggm.regression_model_helper import generate_noise_for_model_test


def norm_log_likelihood(
    x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor
) -> torch.Tensor:
    norm = D.Independent(D.Normal(mean, torch.sqrt(var)), 1)
    return norm.log_prob(x)


class MCDRegressor(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        out_dim: int,
        activation: str,
        dropout_rate: float = mcd_regressor_parameters[DROPOUT_RATE].default,
        eps: float = mcd_regressor_parameters[EPS].default,
        learning_rate: float = mcd_regressor_parameters[LEARNING_RATE].default,
        n_mc_samples: int = mcd_regressor_parameters[N_MC_SAMPLES].default,
    ) -> None:
        super(MCDRegressor, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.activation = activation

        f = get_activation_function(activation)
        self.μ = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(p=dropout_rate),
            f,
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(p=dropout_rate),
        )

        self.dropout_rate = dropout_rate
        self.n_mc_samples = n_mc_samples

        self.learning_rate = learning_rate

        self.eps = eps

        self.example_input_array = torch.rand((10, self.input_dim))

        self.save_hyperparameters(
            "input_dim",
            "hidden_dim",
            "out_dim",
            "learning_rate",
            "activation",
            "dropout_rate",
            "n_mc_samples",
        )

    def _generate_mc_samples(self, x: torch.Tensor) -> torch.Tensor:
        prev_training = copy(self.training)
        # Need dropout to be activated
        self.train()
        y_mc_samples = torch.zeros(self.n_mc_samples, x.shape[0], self.out_dim).type_as(
            x
        )
        for i in range(self.n_mc_samples):
            y_mc_samples[i, :] = self.μ(x)
        self.train(mode=prev_training)
        return y_mc_samples

    def _predictive_mean(self, y_mc_samples: torch.Tensor) -> torch.Tensor:
        return y_mc_samples.mean(dim=0)

    def _predictive_std(self, y_mc_samples: torch.Tensor) -> torch.Tensor:
        return y_mc_samples.std(dim=0) + self.eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_mc_samples = self._generate_mc_samples(x)
        μ_x = self._predictive_mean(y_mc_samples)
        σ_x = self._predictive_std(y_mc_samples)
        return (μ_x, σ_x)

    def predictive_mean(self, x: torch.Tensor, *args) -> torch.Tensor:
        y_mc_samples = self._generate_mc_samples(x)
        μ_x = self._predictive_mean(y_mc_samples)
        return μ_x

    def predictive_std(self, x: torch.Tensor, *args) -> torch.Tensor:
        y_mc_samples = self._generate_mc_samples(x)
        σ_x = self._predictive_std(y_mc_samples)
        return σ_x

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        μ_x = self.μ(x)
        loss = F.mse_loss(μ_x, y)
        self.log(TRAIN_LOSS, loss, on_epoch=True)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        μ_x = self.μ(x)
        loss = F.mse_loss(μ_x, y)
        self.log(EVAL_LOSS, loss, on_epoch=True)
        return loss

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        y_pred = self.predictive_mean(x)
        var_pred = self.predictive_std(x) ** 2
        empirical_var = (y_pred - y) ** 2

        loss = F.mse_loss(y_pred, y)
        self.log(TEST_LOSS, loss, on_epoch=True)

        # LLK
        log_likelihood = norm_log_likelihood(y, y_pred, var_pred)
        self.log(TEST_ELLK, torch.mean(log_likelihood), on_epoch=True)

        # Mean fit
        self.log(TEST_MEAN_FIT_MAE, F.l1_loss(y_pred, y), on_epoch=True)
        self.log(TEST_MEAN_FIT_RMSE, torch.sqrt(loss), on_epoch=True)

        # Variance fit
        self.log(
            TEST_VARIANCE_FIT_MAE, F.l1_loss(var_pred, empirical_var), on_epoch=True
        )
        self.log(
            TEST_VARIANCE_FIT_RMSE,
            torch.sqrt(F.mse_loss(var_pred, empirical_var)),
            on_epoch=True,
        )

        # Noise
        x_noisy = generate_noise_for_model_test(x)
        self.log(
            NOISE_UNCERTAINTY, torch.mean(self.predictive_std(x_noisy)), on_epoch=True
        )

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(
            self.μ.parameters(),
            lr=self.learning_rate,
        )
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        return model_specific_args(mcd_regressor_parameters, parent_parser)


class ENSRegressor(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        out_dim: int,
        activation: str,
        learning_rate: float = ens_regressor_parameters[LEARNING_RATE].default,
        eps: float = ens_regressor_parameters[EPS].default,
        n_ens: int = ens_regressor_parameters[N_ENS].default,
    ) -> None:
        super(ENSRegressor, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.eps = eps
        self.n_ens = n_ens

        self.std = 0.02

        self.activation = activation

        f = get_activation_function(activation)

        for i in range(self.n_ens):
            setattr(
                self,
                f"μ_{i}",
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    f,
                    nn.Linear(hidden_dim, out_dim),
                ),
            )

        for i in range(self.n_ens):
            setattr(
                self,
                f"σ_{i}",
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    f,
                    nn.Linear(hidden_dim, out_dim),
                    nn.Softplus(),
                ),
            )

        self.learning_rate = learning_rate

        self.example_input_array = torch.rand((10, self.input_dim))

        self.save_hyperparameters(
            "input_dim",
            "hidden_dim",
            "out_dim",
            "learning_rate",
            "activation",
            "n_ens",
        )

    @property
    def automatic_optimization(self) -> bool:
        return False

    @property
    def μ(self) -> List[nn.Module]:
        return [getattr(self, f"μ_{i}") for i in range(self.n_ens)]

    @property
    def σ(self) -> List[nn.Module]:
        return [getattr(self, f"σ_{i}") for i in range(self.n_ens)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _μ_x = torch.cat([_μ(x) for _μ in self.μ], dim=1)
        _σ_x = torch.cat([_σ(x) for _σ in self.σ], dim=1) + self.eps

        μ_x = _μ_x.mean(dim=1)[:, None]

        σ_x = (_σ_x + _μ_x ** 2).mean(dim=1)[:, None] - μ_x ** 2

        return (μ_x, σ_x)

    def predictive_mean(self, x: torch.Tensor, *args) -> torch.Tensor:
        return self(x)[0]

    def predictive_std(self, x: torch.Tensor, *args) -> torch.Tensor:
        return self(x)[1]

    def _update_hacks(self):
        self.switch = 1 if self.current_epoch > self.trainer.max_epochs / 2 else 0

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        optimizer_idx: int = None,
    ) -> torch.Tensor:
        optimizers: Union[
            List[torch.optim.Optimizer], torch.optim.Optimizer
        ] = self.optimizers()
        if isinstance(optimizers, torch.optim.Optimizer):
            optimizers = [optimizers]

        x, y = batch

        self._update_hacks()

        loss = torch.zeros((self.n_ens,)).type_as(x)
        for idx_ens in range(self.n_ens):
            μ_x, σ_x = self.μ[idx_ens](x), self.σ[idx_ens](x) + self.eps

            if not self.switch:
                σ_x = self.std * torch.ones_like(σ_x)

            _loss = -norm_log_likelihood(y, μ_x, σ_x ** 2).mean()

            _opt = optimizers[idx_ens]
            _opt.zero_grad()
            self.manual_backward(_loss, _opt)
            _opt.step()

            loss[idx_ens] = _loss

        loss = loss.mean()
        self.log(TRAIN_LOSS, loss, on_epoch=True)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        μ_x, σ_x = self(x)
        loss = norm_log_likelihood(y, μ_x, σ_x ** 2).sum()
        self.log(EVAL_LOSS, loss, on_epoch=True)
        return loss

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        μ_x, σ_x = self(x)
        log_likelihood = norm_log_likelihood(y, μ_x, σ_x ** 2)

        loss = log_likelihood.sum()
        self.log(TEST_LOSS, loss, on_epoch=True)

        self.log(TEST_ELLK, torch.mean(log_likelihood), on_epoch=True)

        y_pred = μ_x
        var_pred = σ_x ** 2
        empirical_var = (y_pred - y) ** 2

        # Mean fit
        self.log(TEST_MEAN_FIT_MAE, F.l1_loss(y_pred, y), on_epoch=True)
        self.log(TEST_MEAN_FIT_RMSE, torch.sqrt(loss), on_epoch=True)

        # Variance fit
        self.log(
            TEST_VARIANCE_FIT_MAE, F.l1_loss(var_pred, empirical_var), on_epoch=True
        )
        self.log(
            TEST_VARIANCE_FIT_RMSE,
            torch.sqrt(F.mse_loss(var_pred, empirical_var)),
            on_epoch=True,
        )

        # Noise
        x_noisy = generate_noise_for_model_test(x)
        self.log(
            NOISE_UNCERTAINTY, torch.mean(self.predictive_std(x_noisy)), on_epoch=True
        )

        return loss

    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        optimizers = [
            torch.optim.Adam(
                chain(self.μ[i].parameters(), self.σ[i].parameters()),
                lr=self.learning_rate,
            )
            for i in range(self.n_ens)
        ]
        return optimizers

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        return model_specific_args(ens_regressor_parameters, parent_parser)
