import torch
from torch import nn
import math
from miro.models.networks.positional_embeddings import FourierEmbedding


class FlowPrecond(nn.Module):
    """Flow preconditioning for RIN-style networks with latent self-conditioning.

    Features:
    - Input/output scaling (c_in, c_out) with noise-driven normalization
    - Logvar projection for gradient norm reweighting
    - Latent self-conditioning (previous_latents initialization)
    """

    def __init__(
        self,
        num_latents=256,
        latents_dim=512,
        do_normalization="noise_driven",
        sigma_data=0.5,
        do_gradnorm_reweighting=True,
        logvar_channels=128,
        logvar_mlp_layers=0,
    ):
        super().__init__()
        self.num_latents = num_latents
        self.latent_dim = latents_dim
        self.do_normalization = do_normalization
        self.sigma_data = sigma_data
        self.do_gradnorm_reweighting = do_gradnorm_reweighting
        self.logvar_channels = logvar_channels
        self.logvar_mlp_layers = logvar_mlp_layers

        if do_gradnorm_reweighting:
            self._build_logvar_modules()

    def _build_logvar_modules(self):
        def _make_mlp_layers():
            layers = []
            for _ in range(self.logvar_mlp_layers):
                layers += [nn.Linear(self.logvar_channels, self.logvar_channels), nn.SiLU()]
            return layers

        input_layer = [FourierEmbedding(self.logvar_channels)]
        self.logvar_proj = nn.Sequential(
            *input_layer,
            *_make_mlp_layers(),
            nn.Linear(self.logvar_channels, 1),
        )

    def _compute_scaling(self, batch):
        gamma = batch["gamma"]
        if self.do_normalization == "noise_driven" or self.do_normalization is True:
            c_in = 1 / torch.sqrt(
                (gamma * self.sigma_data) ** 2 + (1 - gamma) ** 2
            ).reshape(-1, *[1] * (batch["y"].ndim - 1))
            c_out = torch.ones_like(c_in) * math.sqrt(1 + self.sigma_data**2)
        else:
            ones_shape = (batch["y"].shape[0],) + (1,) * (batch["y"].ndim - 1)
            c_in = torch.ones(ones_shape, device=batch["y"].device, dtype=batch["y"].dtype)
            c_out = torch.ones_like(c_in)
        return c_in, c_out

    def _compute_logvar(self, batch):
        gamma = batch["gamma"].flatten()
        return self.logvar_proj(gamma)

    def compute_logvar(self, batch):
        self._prepare_batch(batch)
        return self._compute_logvar(batch)

    def _prepare_batch(self, batch):
        if "previous_latents" not in batch or batch["previous_latents"] is None:
            batch["previous_latents"] = torch.zeros(
                batch["y"].shape[0],
                self.num_latents,
                self.latent_dim,
                device=batch["y"].device,
                dtype=torch.float32,
            )

    def forward(self, network, batch, return_logvar=False):
        c_in, c_out = self._compute_scaling(batch)
        self._prepare_batch(batch)

        batch["y"] = batch["y"] * c_in
        F_x, z = network(batch)
        F_x = F_x * c_out

        output = [F_x, z]
        if return_logvar:
            logvar = self._compute_logvar(batch)
            output.append(logvar.reshape(-1, *[1] * (batch["y"].ndim - 1)))
        return tuple(output)
