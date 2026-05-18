"""High-level inference pipeline for MIRO text-to-image diffusion models.

Mirrors the ergonomics of ``diffusers.StableDiffusionPipeline`` but stays a
thin orchestrator around the existing ``miro.models.*`` building blocks.

Example:
    >>> from miro import MiroPipeline
    >>> pipe = MiroPipeline.from_pretrained("nicolas-dufour/miro").to("cuda")
    >>> image = pipe(
    ...     "Photography closeup portrait of an adorable rusty broken­down "
    ...     "steampunk robot covered in budding vegetation, surrounded by tall "
    ...     "grass, misty futuristic sci­fi forest environment.",
    ...     num_inference_steps=50,
    ... )[0]
    >>> image.save("out.png")
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from miro.models.networks.rin import CADRINTextCond
from miro.models.postprocessing import VAEDecoderPostProcessing
from miro.models.preconditioning import FlowPrecond
from miro.models.preprocessing import PrecomputedSDLatentPreconditioning
from miro.models.sampler import flow_euler_sampler
from miro.models.scheduler import LinearScheduler

DEFAULT_TEXT_ENCODER = "google/flan-t5-xl"
DEFAULT_VAE = "stabilityai/sdxl-vae"

COHERENCE_KEYS = (
    "clip_score",
    "aesthetic_score",
    "image_reward_score",
    "pick_a_score_score",
    "hpsv2_score",
    "vqa_score",
    "sciscore_score",
)

_PIPELINE_FILES = ("config.json", "model.safetensors", "uncond_embedding.npy")


def _strip_prefix(state_dict: Mapping[str, torch.Tensor], prefix: str) -> dict:
    p = prefix + "."
    return {k[len(p):]: v for k, v in state_dict.items() if k.startswith(p)}


class MiroPipeline(nn.Module):
    """Text-to-image pipeline for MIRO checkpoints.

    The pipeline owns the diffusion network (EMA weights), preconditioning, the
    VAE decoder, the noise scheduler, and a precomputed FLAN-T5-XL unconditional
    embedding. The T5 tokenizer + encoder are loaded lazily on the first call
    so the pipeline is cheap to construct.
    """

    COHERENCE_KEYS = COHERENCE_KEYS

    def __init__(
        self,
        network: CADRINTextCond,
        preconditioning: FlowPrecond,
        data_preprocessing: PrecomputedSDLatentPreconditioning,
        postprocessing: VAEDecoderPostProcessing,
        scheduler: LinearScheduler,
        uncond_embedding: torch.Tensor,
        config: dict,
        text_encoder=None,
        tokenizer=None,
    ):
        super().__init__()
        self.network = network.eval()
        self.preconditioning = preconditioning.eval()
        self.data_preprocessing = data_preprocessing.eval()
        self.postprocessing = postprocessing.eval()
        self.scheduler = scheduler
        self.register_buffer("uncond_embedding", uncond_embedding, persistent=False)
        self.config = config
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self._device = torch.device("cpu")
        self._dtype = torch.float32

    # ------------------------------------------------------------------ #
    # Loading                                                            #
    # ------------------------------------------------------------------ #
    @classmethod
    def from_pretrained(
        cls,
        repo_id_or_path: str,
        *,
        variant: str | None = None,
        cache_dir: str | None = None,
        revision: str | None = None,
        torch_dtype: torch.dtype | None = None,
        device: str | torch.device | None = None,
        text_encoder: str | None = None,
    ) -> "MiroPipeline":
        """Load a pipeline from a HuggingFace repo id or a local directory.

        ``variant`` selects a subfolder (used by the ablations repo, e.g.
        ``MiroPipeline.from_pretrained("nicolas-dufour/miro-ablations", variant="no_clip")``).
        """
        local_root = Path(repo_id_or_path)
        if local_root.exists() and local_root.is_dir():
            base = local_root / variant if variant else local_root
            files = {name: base / name for name in _PIPELINE_FILES}
            for name, p in files.items():
                if not p.exists():
                    raise FileNotFoundError(f"Missing {name} in {base}")
        else:
            from huggingface_hub import hf_hub_download

            kwargs = {
                "repo_id": repo_id_or_path,
                "cache_dir": cache_dir,
                "revision": revision,
            }
            if variant:
                kwargs["subfolder"] = variant
            files = {
                name: Path(hf_hub_download(filename=name, **kwargs))
                for name in _PIPELINE_FILES
            }

        with open(files["config.json"]) as f:
            config = json.load(f)

        network = CADRINTextCond(**config["network"])
        preconditioning = FlowPrecond(**config["preconditioning"])
        data_preprocessing = PrecomputedSDLatentPreconditioning(**config["data_preprocessing"])

        from diffusers import AutoencoderKL

        vae = AutoencoderKL.from_pretrained(DEFAULT_VAE)
        postprocessing = VAEDecoderPostProcessing(vae=vae, **config["postprocessing"])

        scheduler = LinearScheduler(**config.get("scheduler", {"start": 1, "end": 0, "clip_min": 1e-9}))

        from safetensors.torch import load_file

        state = load_file(str(files["model.safetensors"]))
        network.load_state_dict(_strip_prefix(state, "network"), strict=True)
        preconditioning.load_state_dict(_strip_prefix(state, "preconditioning"), strict=True)
        data_preprocessing.load_state_dict(
            _strip_prefix(state, "data_preprocessing"), strict=False
        )
        post_state = _strip_prefix(state, "postprocessing")
        # The VAE weights are loaded from `DEFAULT_VAE`; only the normalization
        # buffers (scale/bias) live in our safetensors file.
        post_state = {k: v for k, v in post_state.items() if not k.startswith("vae.")}
        postprocessing.load_state_dict(post_state, strict=False)

        uncond = np.load(files["uncond_embedding.npy"])
        uncond_embedding = torch.from_numpy(uncond.astype(np.float32))

        pipe = cls(
            network=network,
            preconditioning=preconditioning,
            data_preprocessing=data_preprocessing,
            postprocessing=postprocessing,
            scheduler=scheduler,
            uncond_embedding=uncond_embedding,
            config=config,
        )
        pipe._text_encoder_name = text_encoder or DEFAULT_TEXT_ENCODER

        if device is not None or torch_dtype is not None:
            pipe = pipe.to(device=device, dtype=torch_dtype)
        return pipe

    # ------------------------------------------------------------------ #
    # Device / dtype management                                          #
    # ------------------------------------------------------------------ #
    def to(self, *args, **kwargs):  # type: ignore[override]
        device, dtype, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self._device = torch.device(device)
        if dtype is not None:
            self._dtype = dtype
        super().to(device=device, dtype=dtype)
        if self.text_encoder is not None and device is not None:
            self.text_encoder.to(device)
        return self

    # ------------------------------------------------------------------ #
    # Text encoding (lazy)                                               #
    # ------------------------------------------------------------------ #
    def _ensure_text_encoder(self):
        if self.text_encoder is not None and self.tokenizer is not None:
            return
        from transformers import AutoTokenizer, T5EncoderModel

        name = getattr(self, "_text_encoder_name", DEFAULT_TEXT_ENCODER)
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        encoder = T5EncoderModel.from_pretrained(name).eval()
        encoder.requires_grad_(False)
        encoder.to(self._device)
        self.text_encoder = encoder

    def _encode_prompts(
        self, prompts: list[str], num_images_per_prompt: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = self.config.get("max_text_len", 77)
        tok = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            max_length=max_len,
            truncation=True,
        )
        input_ids = tok["input_ids"].to(self._device)
        attention_mask = tok["attention_mask"].to(self._device)
        with torch.no_grad():
            emb = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
        emb = emb.float()
        mask = attention_mask.bool()
        if num_images_per_prompt > 1:
            emb = emb.repeat_interleave(num_images_per_prompt, dim=0)
            mask = mask.repeat_interleave(num_images_per_prompt, dim=0)
        return emb, mask

    # ------------------------------------------------------------------ #
    # Reward conditioning                                                #
    # ------------------------------------------------------------------ #
    @property
    def coherence_keys(self) -> tuple[str, ...]:
        """Reward axes this checkpoint was trained on (from ``config.json``).

        Returns the full 7-key tuple for main MIRO and shorter subsets for
        single-reward specialists / leave-one-out ablations.
        """
        keys = self.config.get("coherence_keys")
        return tuple(keys) if keys else self.COHERENCE_KEYS

    def _build_coherence(
        self,
        targets: Mapping[str, float] | None,
        batch_size: int,
        default: float,
        suffix: str,
    ) -> dict[str, torch.Tensor]:
        targets = dict(targets) if targets else {}
        valid = set(self.coherence_keys)
        unknown = set(targets) - valid
        if unknown:
            raise ValueError(
                f"reward_targets contains keys this checkpoint was not trained on: "
                f"{sorted(unknown)}. Valid keys: {sorted(valid)}."
            )
        out = {}
        for base in self.coherence_keys:
            value = float(targets.get(base, default))
            out[f"{base}_{suffix}"] = torch.full(
                (batch_size,), value, device=self._device, dtype=torch.float32
            )
        return out

    # ------------------------------------------------------------------ #
    # Inference                                                          #
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def __call__(
        self,
        prompt: str | list[str],
        *,
        reward_targets: Mapping[str, float] | None = None,
        negative_reward_targets: Mapping[str, float] | None = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.0,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | None = None,
        output_type: str = "pil",
    ) -> list[Image.Image] | np.ndarray:
        self._ensure_text_encoder()
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        text_emb, text_mask = self._encode_prompts(prompts, num_images_per_prompt)
        B = text_emb.shape[0]

        coh = self._build_coherence(reward_targets, B, default=1.0, suffix="coherence")
        uncoh = self._build_coherence(
            negative_reward_targets, B, default=0.0, suffix="coherence"
        )

        res = int(self.config["data_resolution"])
        x_N = torch.randn(
            B, 4, res, res,
            device=self._device, dtype=torch.float32, generator=generator,
        )

        batch = {
            "y": x_N,
            "previous_latents": None,
            "text_tokens_embeddings": text_emb,
            "text_tokens_mask": text_mask,
        }
        batch.update(coh)

        uncond_emb = self.uncond_embedding.to(self._device)
        if uncond_emb.dim() == 1:
            uncond_emb = uncond_emb.unsqueeze(0)
        uncond_emb_b = uncond_emb.unsqueeze(0).expand(B, -1, -1).contiguous()
        uncond_mask_b = torch.ones(
            B, uncond_emb_b.shape[1], dtype=torch.bool, device=self._device
        )
        uncond_tokens = {
            "text_tokens_embeddings": uncond_emb_b,
            "text_tokens_mask": uncond_mask_b,
        }

        def ema_model(b):
            return self.preconditioning(self.network, b)

        coherence_keys_full = [f"{k}_coherence" for k in self.coherence_keys]
        coherence_values = {k: float(coh[k][0].item()) for k in coherence_keys_full}
        uncoherence_values = {k: float(uncoh[k][0].item()) for k in coherence_keys_full}

        sigma_data = float(self.config.get("preconditioning", {}).get("sigma_data", 0.5))

        autocast_enabled = self._dtype != torch.float32 and self._device.type == "cuda"
        with torch.amp.autocast(
            device_type=self._device.type if autocast_enabled else "cpu",
            dtype=self._dtype if autocast_enabled else torch.float32,
            enabled=autocast_enabled,
        ):
            latents = flow_euler_sampler(
                ema_model,
                copy.copy(batch),
                num_steps=num_inference_steps,
                scheduler=self.scheduler,
                conditioning_keys=["text_tokens"],
                uncond_tokens=uncond_tokens,
                cfg_rate=guidance_scale,
                generator=generator,
                coherence_keys=coherence_keys_full,
                coherence_values=coherence_values,
                uncoherence_values=uncoherence_values,
                sigma_data=sigma_data,
                data_mean=0.5,
                data_std=0.5,
            )

        images = self.postprocessing(latents.to(self._dtype if self._dtype != torch.float32 else torch.float32))
        if output_type == "numpy":
            return images
        return [Image.fromarray(img) for img in images]
