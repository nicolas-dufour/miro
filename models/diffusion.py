import copy
import logging
from typing import Any

import numpy as np
import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from hydra.utils import instantiate

from miro.models.networks.transformers import RMSNorm
from miro.utils.misc import print0


def gradient_sanity_check(model):
    """Assert that gradient norms are identical across all DDP ranks."""
    if not isinstance(model, DistributedDataParallel):
        print0("Model is not a DDP module, skipping gradient sanity check")
        return
    torch.cuda.synchronize()
    for name, p in model.module.named_parameters():
        if p.requires_grad and p.grad is not None and len(p.shape) > 3:
            monitor = p.grad.norm()
            monitor_list = [torch.zeros_like(monitor) for _ in range(dist.get_world_size())]
            dist.all_gather(monitor_list, monitor)
            monitor_tensor = torch.stack(monitor_list)
            ref = monitor_tensor[0]
            for i, m in enumerate(monitor_tensor):
                assert torch.isclose(m, ref), (
                    f"Gradient norm mismatch for '{name}' at rank {i}: {m} vs rank 0: {ref}"
                )
            break
    print0("Gradient norm sanity check passed")


def _decay_to_attr_suffix(decay: float) -> str:
    return "_" + str(decay).replace(".", "_")


def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    result += list(model._parameters.keys())
    return result


class DiffusionModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.strict_loading = False
        self.cfg = cfg
        self.network = instantiate(cfg.network.instance)
        if cfg.get("compile", True):
            print0("Using Compiled Model")
            self.network.compile(fullgraph=True)

        self.train_noise_scheduler = instantiate(cfg.train_noise_scheduler)
        self.inference_noise_scheduler = instantiate(cfg.inference_noise_scheduler)
        self.data_preprocessing = instantiate(cfg.data_preprocessing)
        self.cond_preprocessing = instantiate(cfg.cond_preprocessing)
        self.preconditioning = instantiate(cfg.preconditioning)

        self._create_ema_networks(cfg)
        self.postprocessing = instantiate(cfg.postprocessing)
        self.val_sampler = instantiate(cfg.val_sampler)
        self.test_sampler = instantiate(cfg.test_sampler)

        uncond_conditioning = instantiate(cfg.uncond_conditioning)
        if isinstance(uncond_conditioning, np.ndarray):
            self.uncond_conditioning = nn.Parameter(
                torch.from_numpy(uncond_conditioning), requires_grad=False
            )
        else:
            self.uncond_conditioning = uncond_conditioning

        self.loss = instantiate(cfg.loss)(
            self.train_noise_scheduler,
            self.uncond_conditioning,
        )

    def _create_ema_networks(self, cfg):
        ema_decay = cfg.get("ema_decay", 0.9999)
        if isinstance(ema_decay, (int, float)):
            decays = [float(ema_decay)]
        else:
            decays = [float(d) for d in ema_decay]

        default_ema_index = cfg.get("default_ema_index", 0)
        if default_ema_index < 0 or default_ema_index >= len(decays):
            default_ema_index = 0

        self._ema_decays = decays
        self._default_ema_index = default_ema_index

        self._ema_attr_names = []
        for i, decay in enumerate(decays):
            if i == 0:
                attr_name = "ema_network"
            else:
                attr_name = "ema_network" + _decay_to_attr_suffix(decay)
            self._ema_attr_names.append(attr_name)

        for i, decay in enumerate(decays):
            ema_net = copy.deepcopy(self.network).requires_grad_(False)
            ema_net.eval()
            self.add_module(self._ema_attr_names[i], ema_net)

    def get_default_ema_network(self):
        attr_name = self._ema_attr_names[self._default_ema_index]
        return getattr(self, attr_name)

    def training_step(self, batch, batch_idx):
        with torch.no_grad():
            batch = self.data_preprocessing(batch)
            batch = self.cond_preprocessing(batch)
        batch_size = batch["x_0"].shape[0]
        loss, losses = self.loss(
            self.preconditioning,
            self.network,
            self.ema_network,
            batch,
            global_step=self.global_step,
            generator=getattr(self, "training_generator", None),
        )
        loss = loss.mean()
        self.log(
            "train/loss",
            loss,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        for key, value in losses.items():
            self.log(
                f"train/{key}",
                value.mean(),
                sync_dist=True,
                on_step=True,
                on_epoch=True,
                batch_size=batch_size,
            )
        return loss

    def on_before_optimizer_step(self, optimizer):
        if self.global_step == 0:
            no_grad = []
            for name, param in self.network.named_parameters():
                if param.grad is None and "dummy" not in name:
                    no_grad.append(name)
            if len(no_grad) > 0:
                print0("Parameters without grad:")
                print0(no_grad)

        if self.global_step < 5:
            gradient_sanity_check(self.trainer.model)

    def on_train_start(self):
        self.training_generator = None

    def on_validation_start(self):
        self.validation_generator = torch.Generator(device=self.device).manual_seed(
            3407
        )
        self.validation_generator_ema = torch.Generator(device=self.device).manual_seed(
            3407
        )

    def validation_step(self, batch, batch_idx):
        batch = self.data_preprocessing(batch)
        batch = self.cond_preprocessing(batch)
        batch_size = batch["x_0"].shape[0]

        loss, losses = self.loss(
            self.preconditioning,
            self.network,
            self.ema_network,
            batch,
            global_step=self.global_step,
            generator=self.validation_generator,
        )
        loss = loss.mean()
        self.log(
            "val/loss",
            loss,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        for key, value in losses.items():
            self.log(
                f"val/{key}",
                value.mean(),
                sync_dist=True,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )

        if hasattr(self, "ema_model"):
            loss_ema, losses_ema = self.loss(
                self.preconditioning,
                self.get_default_ema_network(),
                self.get_default_ema_network(),
                batch,
                global_step=self.global_step,
                generator=self.validation_generator_ema,
            )
            loss_ema = loss_ema.mean()
            self.log(
                "val/loss_ema",
                loss_ema,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
            for key, value in losses_ema.items():
                self.log(
                    f"val/{key}_ema",
                    value.mean(),
                    sync_dist=True,
                    on_step=False,
                    on_epoch=True,
                    batch_size=batch_size,
                )

    def on_train_batch_end(self, outputs, batch, batch_idx):
        super().on_train_batch_end(outputs, batch, batch_idx)

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        if self.cfg.optimizer.exclude_ln_and_biases_from_weight_decay:
            parameters_names_wd = get_parameter_names(
                self.network, [nn.LayerNorm, nn.Embedding, RMSNorm]
            )
            parameters_names_wd = [
                name for name in parameters_names_wd if "bias" not in name
            ]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in self.network.named_parameters()
                        if n in parameters_names_wd
                    ]
                    + list(self.preconditioning.parameters()),
                    "weight_decay": self.cfg.optimizer.optim.weight_decay,
                    "layer_adaptation": True,
                },
                {
                    "params": [
                        p
                        for n, p in self.network.named_parameters()
                        if n not in parameters_names_wd
                    ],
                    "weight_decay": 0.0,
                    "layer_adaptation": False,
                },
            ]
            optimizer = instantiate(
                self.cfg.optimizer.optim, optimizer_grouped_parameters
            )
        else:
            optimizer = instantiate(
                self.cfg.optimizer.optim,
                self.network.parameters() + list(self.preconditioning.parameters()),
            )
        scheduler = instantiate(self.cfg.lr_scheduler)(optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.global_step)

    def sample(
        self,
        batch_size=None,
        shape=None,
        cond=None,
        x_N=None,
        latents=None,
        num_steps=None,
        sampler=None,
        scheduler=None,
        stage="test",
        cfg=0,
        guidance_type="constant",
        guidance_start_step=0,
        guidance_end_step=None,
        generator=None,
        coherence_values={"clip_score_coherence": 1.0},
        uncoherence_values={"clip_score_uncoherence": 0.0},
        unconfident_prompt=None,
        coherence_keys: list[str] | None = None,
    ):
        batch = {"previous_latents": latents}
        if x_N is None and (shape is None or batch_size is None):
            raise ValueError("Shape must be specified if x_N are not provided")
        if x_N is None:
            x_N = torch.randn(batch_size, *shape, device=self.device, generator=generator)
        else:
            if x_N.ndim == 3:
                x_N = x_N.unsqueeze(0)
            batch_size = x_N.shape[0]
            shape = x_N.shape[1:]
        batch["y"] = x_N
        if sampler is None:
            if stage == "val":
                sampler = self.val_sampler
            elif stage == "test":
                sampler = self.test_sampler
            else:
                raise ValueError(f"Unknown stage {stage}")
        if scheduler is None:
            scheduler = self.inference_noise_scheduler
        if unconfident_prompt is not None:
            uncond_conditioning = unconfident_prompt
        else:
            uncond_conditioning = self.uncond_conditioning

        if cond is not None:
            if isinstance(cond, dict):
                for key, value in cond.items():
                    batch[key] = value
                if isinstance(uncond_conditioning, torch.Tensor):
                    uncond_tokens = uncond_conditioning
                    if uncond_tokens.ndim == 2:
                        b = batch_size if batch_size is not None else 1
                        uncond_tokens = uncond_tokens.unsqueeze(0).repeat(b, 1, 1)
                    elif uncond_tokens.ndim == 3:
                        if batch_size is not None and uncond_tokens.shape[0] != batch_size:
                            if uncond_tokens.shape[0] == 1:
                                uncond_tokens = uncond_tokens.repeat(batch_size, 1, 1)
                            else:
                                raise ValueError(
                                    f"unconfident_prompt batch size {uncond_tokens.shape[0]} does not match batch_size {batch_size}"
                                )
                    uncond_tokens_mask = torch.ones(
                        (uncond_tokens.shape[0], uncond_tokens.shape[1]),
                        dtype=torch.bool,
                        device=uncond_tokens.device,
                    )
                    uncond_tokens_batch = {
                        f"{self.cfg.cond_preprocessing.input_key}_embeddings": uncond_tokens,
                        f"{self.cfg.cond_preprocessing.input_key}_mask": uncond_tokens_mask,
                    }
                else:
                    uncond_tokens = uncond_conditioning
                    if isinstance(uncond_tokens, str):
                        uncond_tokens = [uncond_tokens] * (
                            1 if batch_size is None else batch_size
                        )
                    uncond_tokens_batch = {
                        self.cfg.cond_preprocessing.input_key: uncond_tokens
                    }
            else:
                if isinstance(cond, str):
                    uncond_tokens = [uncond_conditioning] * (
                        1 if batch_size is None else batch_size
                    )
                    cond = [cond] * (1 if batch_size is None else batch_size)
                elif isinstance(cond, torch.Tensor) and isinstance(
                    self.uncond_conditioning, float
                ):
                    uncond_tokens = (
                        torch.ones_like(cond, device=self.device) * uncond_conditioning
                    )
                else:
                    if len(uncond_conditioning.shape) < len(cond.shape):
                        uncond_tokens = uncond_conditioning.repeat(
                            1 if batch_size is None else batch_size,
                            *[1 for _ in uncond_conditioning.shape],
                        )
                    else:
                        uncond_tokens = uncond_conditioning
                batch[self.cfg.cond_preprocessing.input_key] = cond
                uncond_tokens_batch = {
                    self.cfg.cond_preprocessing.input_key: uncond_tokens
                }
            uncond_tokens = self.cond_preprocessing(
                uncond_tokens_batch,
                device=self.device,
            )
        else:
            uncond_tokens = None

        batch = self.cond_preprocessing(batch, device=self.device)
        if num_steps is None:
            image = sampler(
                self.ema_model,
                batch,
                conditioning_keys=[self.cfg.cond_preprocessing.output_key_root],
                scheduler=scheduler,
                uncond_tokens=uncond_tokens,
                cfg_rate=cfg,
                guidance_type=guidance_type,
                guidance_start_step=guidance_start_step,
                guidance_end_step=guidance_end_step,
                generator=generator,
                coherence_keys=coherence_keys,
                coherence_values=coherence_values,
                uncoherence_values=uncoherence_values,
                sigma_data=self.cfg.sigma_data,
                data_mean=self.cfg.data_mean,
                data_std=self.cfg.data_std,
            )
        else:
            image = sampler(
                self.ema_model,
                batch,
                conditioning_keys=[self.cfg.cond_preprocessing.output_key_root],
                scheduler=scheduler,
                uncond_tokens=uncond_tokens,
                num_steps=num_steps,
                cfg_rate=cfg,
                guidance_type=guidance_type,
                guidance_start_step=guidance_start_step,
                guidance_end_step=guidance_end_step,
                generator=generator,
                coherence_keys=coherence_keys,
                coherence_values=coherence_values,
                uncoherence_values=uncoherence_values,
                sigma_data=self.cfg.sigma_data,
                data_mean=self.cfg.data_mean,
                data_std=self.cfg.data_std,
            )
        return self.postprocessing(image)

    def model(self, *args, **kwargs):
        return self.preconditioning(self.network, *args, **kwargs)

    def ema_model(self, *args, **kwargs):
        return self.preconditioning(self.get_default_ema_network(), *args, **kwargs)
