from typing import Tuple

from geoml import EmbeddedManifold
import geoml.nnj as nnj
import torch
import torch.nn as nn

from sggm.vae_model import VanillaVAE


class VanillaVAEm(VanillaVAE, EmbeddedManifold):
    def __init__(self, *args, **kwargs):
        super(VanillaVAEm, self).__init__(*args, **kwargs)

        self.decoder_μ = self.nn_to_nnj(self.decoder_μ)
        self.decoder_std = self.nn_to_nnj(self.decoder_std)

    @staticmethod
    def _nn_to_nnj(nn_module: nn.Module) -> nn.Module:
        """
        Transform a single nn module to its nnj counterpart
        """
        # Layers
        nnj_module = None
        if isinstance(nn_module, nn.Linear):
            bias = True if nn_module.bias is not None else False
            nnj_module = nnj.Linear(
                nn_module.in_features, nn_module.out_features, bias=bias
            )
        elif isinstance(nn_module, nn.BatchNorm1d):
            nnj_module = nnj.BatchNorm1d(
                nn_module.num_features,
                eps=nn_module.eps,
                momentum=nn_module.momentum,
                affine=nn_module.affine,
                track_running_stats=nn_module.track_running_stats,
            )
        # TODO add support for Conv and BatchNorm
        # Activations
        elif isinstance(nn_module, nn.LeakyReLU):
            nnj_module = nnj.LeakyReLU()
        elif isinstance(nn_module, nn.Sigmoid):
            nnj_module = nnj.Sigmoid()
        elif isinstance(nn_module, nn.Softplus):
            nnj_module = nnj.Softplus()
        else:
            raise NotImplementedError(f"{nn_module} casting to nnj not supported")
        return nnj_module

    @staticmethod
    def nn_to_nnj(mod: nn.Module) -> nn.Module:
        _L = []
        for n_m, m in list(mod.named_children()):
            if isinstance(m, nn.Sequential):
                _L.append(VanillaVAEm.nn_to_nnj(m))
            else:
                _L.append(VanillaVAEm._nn_to_nnj(m))
        return nnj.Sequential(*_L)

    def decoder_jacobian(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, J_μ = self.decoder_μ(z, jacobian=True)
        _, J_σ = self.decoder_std(z, jacobian=True)
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
        # [BS, *self.latent_dims/self.input_size]
        z = z.reshape(-1, self.latent_size)
        μ_z, σ_z = self.decoder_μ(z), self.decoder_std(z)
        # [n_mc_samples, BS, *self.latent_dims/self.input_size]
        μ_z, σ_z = μ_z[None, :], σ_z[None, :]

        # [n_mc_samples, BS, 2 *self.latent_dims/self.input_size]
        embedded = torch.cat((μ_z, σ_z), dim=2)  # BxNx(2D)
        if jacobian:
            J_μ_z, J_σ_z = self.decoder_jacobian(z)
            J_μ_z, J_σ_z = J_μ_z[None, :], J_σ_z[None, :]
            J = torch.cat((J_μ_z, J_σ_z), dim=2)

        if not is_batched:
            embedded = embedded.squeeze(0)

        return embedded
