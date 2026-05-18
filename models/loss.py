from copy import deepcopy
from abc import ABC, abstractmethod

import torch


class BaseLoss(ABC):
    def __init__(
        self,
        scheduler,
        uncond_conditioning,
        conditioning_key="text_tokens",
        self_cond_rate=0.9,
        cond_drop_rate=0.0,
        do_gradnorm_reweighting=False,
    ):
        self.scheduler = scheduler
        self.uncond_conditioning = uncond_conditioning
        self.conditioning_key = conditioning_key
        self.self_cond_rate = self_cond_rate
        self.cond_drop_rate = cond_drop_rate
        self.do_gradnorm_reweighting = do_gradnorm_reweighting

    def _apply_conditional_dropping(self, batch, batch_size, device, generator=None):
        conditioning = None
        conditioning_mask = None
        if f"{self.conditioning_key}_embeddings" in batch:
            conditioning = batch[f"{self.conditioning_key}_embeddings"]
            if f"{self.conditioning_key}_mask" in batch:
                conditioning_mask = batch[f"{self.conditioning_key}_mask"]
        elif self.conditioning_key in batch:
            conditioning = batch[self.conditioning_key]

        if conditioning is not None and self.cond_drop_rate > 0:
            drop_mask = (
                torch.rand(batch_size, device=device, generator=generator)
                < self.cond_drop_rate
            )
            uncond_shape = self.uncond_conditioning.shape
            target_shape = conditioning[drop_mask].shape[1:]
            if uncond_shape == target_shape:
                uncond = self.uncond_conditioning
            elif len(uncond_shape) == 1 and uncond_shape[0] >= target_shape[0]:
                uncond = self.uncond_conditioning[: target_shape[0]]
            else:
                uncond = self.uncond_conditioning.expand_as(conditioning[drop_mask])

            conditioning[drop_mask] = uncond.to(
                conditioning.device, conditioning.dtype
            )

            if conditioning_mask is not None:
                conditioning_mask[drop_mask] = torch.tensor(
                    [1] + [0] * (conditioning_mask.shape[-1] - 1),
                    device=device,
                    dtype=torch.bool,
                )
            batch[self.conditioning_key] = conditioning.detach()
            if conditioning_mask is not None:
                batch[f"{self.conditioning_key}_mask"] = conditioning_mask.detach()

            for key in list(batch.keys()):
                if key.endswith("_coherence"):
                    nan_vals = torch.full_like(batch[key], float("nan"))
                    batch[key] = torch.where(drop_mask, nan_vals, batch[key])
        return batch

    def latent_self_cond(self, preconditioning, network, batch, generator=None):
        output = preconditioning(network, batch)
        latents = output[1]
        drop_mask = (
            torch.rand(
                latents.shape[0],
                *[1 for _ in range(latents.dim() - 1)],
                device=latents.device,
                generator=generator,
            )
            > 1 - self.self_cond_rate
        ).to(latents.dtype)
        latents = latents * drop_mask
        latents = latents.detach()
        network.zero_grad()
        return latents

    def _apply_self_conditioning(
        self, preconditioning, network, batch, x_0, n, t, generator=None
    ):
        batch_self_cond = {k: v.clone() if isinstance(v, torch.Tensor) else deepcopy(v) for k, v in batch.items()}

        # Standard self-conditioning: use same t and n
        y_orig, batch_self_cond = self._calculate_noisy_input(
            x_0, n, t, batch_self_cond, generator
        )
        batch_self_cond["y"] = y_orig

        batch["previous_latents"] = self.latent_self_cond(
            preconditioning,
            network,
            batch_self_cond,
            generator=generator,
        )
        return batch

    @abstractmethod
    def _sample_t(self, batch_size, device, dtype, generator):
        pass

    @abstractmethod
    def _calculate_noisy_input(self, x_0, n, t, batch, generator):
        pass

    @abstractmethod
    def _calculate_loss(self, D_output, x_0, n, t, batch):
        pass

    def __call__(
        self, preconditioning, network, ema_network, batch, global_step, generator=None
    ):
        x_0 = batch["x_0"]
        batch_size = x_0.shape[0]
        device = x_0.device
        dtype = x_0.dtype

        batch = self._apply_conditional_dropping(batch, batch_size, device, generator)

        t = self._sample_t(batch_size, device, dtype, generator)
        n = torch.randn(x_0.shape, device=device, dtype=dtype, generator=generator)

        if self.self_cond_rate > 0:
            batch = self._apply_self_conditioning(
                preconditioning, network, batch, x_0, n, t, generator
            )

        y, batch = self._calculate_noisy_input(x_0, n, t, batch, generator)
        batch["y"] = y

        if self.do_gradnorm_reweighting:
            D_output, _, logvar = preconditioning(
                network, batch, return_logvar=True
            )
        else:
            D_output, _ = preconditioning(network, batch)

        losses = {}
        loss = self._calculate_loss(D_output, x_0, n, t, batch)
        losses["raw_loss"] = loss.mean().detach()
        if self.do_gradnorm_reweighting:
            loss = loss / logvar.exp() + logvar
            losses["loss_gradnorm_reweighted"] = loss.mean().detach()
            losses["logvar"] = logvar.mean().detach()

        return loss, losses


class FlowMatchingLoss(BaseLoss):
    def __init__(self, *args, logit_normal_resample=True, mu_t=0.0, std_t=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.logit_normal_resample = logit_normal_resample
        self.mu_t = mu_t
        self.std_t = std_t

    def _logit_normal_resample(
        self, batch_size, device, generator=None, mu=0.0, std=1.0
    ):
        normal_samples = torch.normal(
            mean=mu, std=std, size=(batch_size,), device=device, generator=generator
        )
        logit_normal_samples = torch.sigmoid(normal_samples)
        return logit_normal_samples

    def _sample_t(self, batch_size, device, dtype, generator, mean=None, std=None):
        if mean is None:
            mean = self.mu_t
        if std is None:
            std = self.std_t
        if self.logit_normal_resample:
            t = self._logit_normal_resample(
                batch_size, device, generator, mean, std
            ).to(dtype)
        else:
            t = torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
        return t

    def _calculate_noisy_input(self, x_0, n, t, batch, generator):
        gamma = self.scheduler(t).reshape(-1, *[1] * (x_0.ndim - 1))
        batch["gamma"] = gamma.reshape(-1)
        y = gamma * x_0 + (1 - gamma) * n
        return y, batch

    def _calculate_loss(self, D_y, x_0, n, t, batch):
        return (D_y - (n - x_0)) ** 2
