import math
from typing import Union
import torch
import torch.distributions as D

from functools import reduce
from numbers import Number
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all

from sggm.definitions import OOD_Z_GENERATION_AVAILABLE_METHODS
from sggm.regression_model_helper import normalise_grad
from sggm.types_ import List


def batch_flatten(x: torch.Tensor) -> torch.Tensor:
    # B, C, H, W -> # B, CxHxW
    return torch.flatten(x, start_dim=1)


def batch_reshape(x: torch.Tensor, shape: List[int]) -> torch.Tensor:
    # B, D -> B, C, H, W where D = CxHxW
    return x.view(-1, *shape)


def reduce_int_list(list: List[int]) -> int:
    return reduce(lambda x, y: x * y, list)


def check_ood_z_generation_method(method: str) -> str:
    if method is None:
        return method
    assert (
        method in OOD_Z_GENERATION_AVAILABLE_METHODS
    ), f"""Method for z ood generation '{method}' is invalid.
    Must either be None or in {OOD_Z_GENERATION_AVAILABLE_METHODS}"""
    return method


def locscale_sigmoid(
    x: torch.Tensor,
    loc: Union[int, float, torch.Tensor],
    scale: Union[int, float, torch.Tensor],
) -> torch.Tensor:
    x = (x - loc) / scale
    return torch.sigmoid(x)


def density_gradient_descent(
    distribution: D.Distribution, x_0: torch.Tensor, params: dict
) -> torch.Tensor:
    N_steps, lr, threshold = params["N_steps"], params["lr"], params["threshold"]

    x_hat = x_0.clone()
    x_hat.requires_grad = True

    print("   PIG gradient descent:", end=" ", flush=True)
    for n in range(N_steps):
        print(f"{n+1}", end=" ")
        with torch.no_grad():
            with torch.set_grad_enabled(True):
                log_prob = distribution.log_prob(x_hat).mean()
                density_grad = torch.autograd.grad(log_prob, x_hat, retain_graph=True)[
                    0
                ]
                normed_density_grad = normalise_grad(density_grad)
                normed_density_grad = torch.where(
                    torch.linalg.norm(density_grad, dim=1)[:, None] < threshold,
                    normed_density_grad,
                    torch.zeros_like(normed_density_grad),
                )
                x_hat = x_hat - lr * normed_density_grad
    print("-> OK")
    x_hat = x_hat.detach()
    return x_hat


CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_INV_SQRT_2 = 1 / math.sqrt(2)
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)


class TruncatedStandardNormal(Distribution):
    """
    Truncated Standard Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    arg_constraints = {
        "a": constraints.real,
        "b": constraints.real,
    }
    has_rsample = True

    def __init__(self, a, b, eps=1e-8, validate_args=None):
        self.a, self.b = broadcast_all(a, b)
        if isinstance(a, Number) and isinstance(b, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.a.size()
        super(TruncatedStandardNormal, self).__init__(
            batch_shape, validate_args=validate_args
        )
        if self.a.dtype != self.b.dtype:
            raise ValueError("Truncation bounds types are different")
        if any(
            (self.a >= self.b)
            .view(
                -1,
            )
            .tolist()
        ):
            raise ValueError("Incorrect truncation range")
        self._dtype_min_gt_0 = torch.tensor(
            torch.finfo(self.a.dtype).eps, dtype=self.a.dtype
        )
        self._dtype_max_lt_1 = torch.tensor(
            1 - torch.finfo(self.a.dtype).eps, dtype=self.a.dtype
        )
        self._little_phi_a = self._little_phi(self.a)
        self._little_phi_b = self._little_phi(self.b)
        self._big_phi_a = self._big_phi(self.a)
        self._big_phi_b = self._big_phi(self.b)
        self._Z = (self._big_phi_b - self._big_phi_a).clamp_min(eps)
        self._log_Z = self._Z.log()
        self._lpbb_m_lpaa_d_Z = (
            self._little_phi_b * self.b - self._little_phi_a * self.a
        ) / self._Z
        self._mean = -(self._little_phi_b - self._little_phi_a) / self._Z
        self._variance = (
            1
            - self._lpbb_m_lpaa_d_Z
            - ((self._little_phi_b - self._little_phi_a) / self._Z) ** 2
        )
        self._entropy = CONST_LOG_SQRT_2PI_E + self._log_Z - 0.5 * self._lpbb_m_lpaa_d_Z

        self._support = constraints.interval(self.a, self.b)

    @property
    def support(self):
        return self._support

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._variance

    @property
    def entropy(self):
        return self._entropy

    @property
    def auc(self):
        return self._Z

    @staticmethod
    def _little_phi(x):
        return (-(x ** 2) * 0.5).exp() * CONST_INV_SQRT_2PI

    @staticmethod
    def _big_phi(x):
        return 0.5 * (1 + (x * CONST_INV_SQRT_2).erf())

    @staticmethod
    def _inv_big_phi(x):
        return CONST_SQRT_2 * (2 * x - 1).erfinv()

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return ((self._big_phi(value) - self._big_phi_a) / self._Z).clamp(0, 1)

    def icdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return self._inv_big_phi(self._big_phi_a + value * self._Z)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return CONST_LOG_INV_SQRT_2PI - self._log_Z - (value ** 2) * 0.5

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        p = torch.empty(shape).uniform_(self._dtype_min_gt_0, self._dtype_max_lt_1)
        return self.icdf(p)

    def expand(self, batch_shape, _instance=None):
        # TODO: it is likely that keeping temporary variables in private attributes violates the logic of this method
        raise NotImplementedError


class TruncatedNormal(TruncatedStandardNormal):
    """
    Truncated Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
        "a": constraints.real,
        "b": constraints.real,
    }
    has_rsample = True

    def __init__(self, loc, scale, a, b, eps=1e-8, validate_args=None):
        self.loc, self.scale, self.a, self.b = broadcast_all(loc, scale, a, b)
        a_standard = (a - self.loc) / self.scale
        b_standard = (b - self.loc) / self.scale
        super(TruncatedNormal, self).__init__(
            a_standard, b_standard, eps=eps, validate_args=validate_args
        )
        self._log_scale = self.scale.log()
        self._mean = self._mean * self.scale + self.loc
        self._variance = self._variance * self.scale ** 2
        self._entropy += self._log_scale

    def _to_std_rv(self, value):
        return (value - self.loc) / self.scale

    def _from_std_rv(self, value):
        return value * self.scale + self.loc

    def cdf(self, value):
        std_rv_value = self._to_std_rv(value)
        if self._validate_args:
            self._validate_sample(std_rv_value)
        return super(TruncatedNormal, self).cdf(std_rv_value)

    def icdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return self._from_std_rv(super(TruncatedNormal, self).icdf(value))

    def log_prob(self, value):
        std_rv_value = self._to_std_rv(value)
        if self._validate_args:
            self._validate_sample(std_rv_value)
        return super(TruncatedNormal, self).log_prob(std_rv_value) - self._log_scale


if __name__ == "__main__":

    #%
    list_test = [1, 2, 3]
    print(f"reduce_int_list: {list_test} -> {reduce_int_list(list_test)}")

    input_shape = (3, 20, 20)
    batch = torch.rand([10, *input_shape])
    batch_flattened = batch_flatten(batch)
    print(f"batch_flatten: {batch.shape} -> {batch_flattened.shape}")

    print(
        f"batch_reshape: {batch_flattened.shape} -> {batch_reshape(batch_flattened, input_shape).shape}"
    )

    from scipy.stats import truncnorm

    loc, scale, a, b = 1.0, 2.0, 1.0, 2.0
    tn_pt = TruncatedNormal(loc, scale, a, b)
    mean_pt, var_pt = tn_pt.mean.item(), tn_pt.variance.item()
    alpha, beta = (a - loc) / scale, (b - loc) / scale
    mean_sp, var_sp = truncnorm.stats(alpha, beta, loc=loc, scale=scale, moments="mv")
    print("mean", mean_pt, mean_sp)
    print("var", var_pt, var_sp)
    print(
        "cdf",
        tn_pt.cdf(1.4).item(),
        truncnorm.cdf(1.4, alpha, beta, loc=loc, scale=scale),
    )
    print(
        "icdf",
        tn_pt.icdf(0.333).item(),
        truncnorm.ppf(0.333, alpha, beta, loc=loc, scale=scale),
    )
    print(
        "logpdf",
        tn_pt.log_prob(1.5).item(),
        truncnorm.logpdf(1.5, alpha, beta, loc=loc, scale=scale),
    )
    print(
        "entropy",
        tn_pt.entropy.item(),
        truncnorm.entropy(alpha, beta, loc=loc, scale=scale),
    )
