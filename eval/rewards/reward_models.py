"""Reward-model wrappers used by score_images.py.

Each scorer is lazy-loaded: instantiating the module is cheap, the heavy
model only loads when ``__call__`` is first invoked. Every scorer takes a
list of PIL images + matching prompts and returns a torch.Tensor of scores
with shape ``(len(images),)``, all detached and on CPU.

Available scorers:

- ``aesthetic``    — Schuhmann sac+logos+ava1-l14-linearMSE MLP on top of
                     OpenAI CLIP ViT-L/14 image embedding.
- ``pick_score``   — yuvalkirstain/PickScore_v1 (CLIP-H based PickScore).
- ``image_reward`` — THUDM ImageReward v1.0.
- ``hpsv2``        — xswu/HPSv2 (ViT-H).
- ``clip_jina``    — jinaai/jina-clip-v2 cosine similarity (* 100).
- ``clip_openai``  — openai/clip-vit-large-patch14 cosine similarity (* 100).

The OpenAI / Pick / HPSv2 / Jina CLIP raw cosine logits are stored on a
[-1, 1]-ish scale after the /100 normalisation.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image


AESTHETIC_DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parent / "models" / "sac+logos+ava1-l14-linearMSE.pth"
)


def _device(d: str | torch.device | None) -> torch.device:
    if d is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(d, torch.device):
        return d
    return torch.device(d)


# ---------------------------------------------------------------------------
# Aesthetic predictor (Schuhmann)
# ---------------------------------------------------------------------------
class _AestheticMLP(torch.nn.Module):
    def __init__(self, input_size: int = 768):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, 1024),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 128),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 16),
            torch.nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


class AestheticScorer:
    name = "aesthetic"

    def __init__(self, device=None, checkpoint: Path | None = None):
        self.device = _device(device)
        self.checkpoint = Path(checkpoint or AESTHETIC_DEFAULT_CHECKPOINT)
        self._mlp = None
        self._clip = None
        self._preprocess = None

    def _load(self) -> None:
        if self._mlp is not None:
            return
        import clip  # OpenAI's clip package

        if not self.checkpoint.exists():
            raise FileNotFoundError(
                f"Aesthetic MLP checkpoint not found at {self.checkpoint}. "
                "Run `uv run miro-rewards-download` first."
            )
        self._mlp = _AestheticMLP(768)
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        self._mlp.load_state_dict(state)
        self._mlp.eval().to(self.device)
        self._clip, self._preprocess = clip.load("ViT-L/14", device=self.device)
        self._clip.eval()

    @torch.no_grad()
    def __call__(self, images: Sequence[Image.Image], prompts: Sequence[str]):
        del prompts
        self._load()
        batch = torch.stack([self._preprocess(img) for img in images]).to(self.device)
        feats = self._clip.encode_image(batch).float()
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return self._mlp(feats).squeeze(-1).detach().cpu()


# ---------------------------------------------------------------------------
# PickScore (yuvalkirstain/PickScore_v1)
# ---------------------------------------------------------------------------
class PickScorer:
    name = "pick_score"

    def __init__(self, device=None):
        self.device = _device(device)
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import CLIPModel, CLIPProcessor

        self._processor = CLIPProcessor.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        )
        self._model = CLIPModel.from_pretrained(
            "yuvalkirstain/PickScore_v1"
        ).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, images: Sequence[Image.Image], prompts: Sequence[str]):
        self._load()
        inputs = self._processor(
            images=list(images),
            text=list(prompts),
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            logits = self._model(**inputs).logits_per_image
        return (torch.diag(logits).float() / 100.0).detach().cpu()


# ---------------------------------------------------------------------------
# ImageReward
# ---------------------------------------------------------------------------
class ImageRewardScorer:
    name = "image_reward"

    def __init__(self, device=None):
        self.device = _device(device)
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import ImageReward as ir
        # Force device for download_root default (CWD) — pinned to ~/.cache.
        self._model = ir.load("ImageReward-v1.0", device=str(self.device))
        self._model.eval()

    @torch.no_grad()
    def __call__(self, images: Sequence[Image.Image], prompts: Sequence[str]):
        # Batched scoring: tokenize all N prompts as a batch, encode all N
        # images as a batch, then run BLIP's text_encoder once with paired
        # cross-attention. Equivalent to N calls of ImageReward.score(p, [img])
        # but ~N× faster.
        self._load()
        model = self._model
        device = next(model.parameters()).device

        text_input = model.blip.tokenizer(
            list(prompts),
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(device)
        # Stack pre-processed images
        image_batch = torch.stack(
            [model.preprocess(img) for img in images]
        ).to(device)

        image_embeds = model.blip.visual_encoder(image_batch)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)
        text_output = model.blip.text_encoder(
            text_input.input_ids,
            attention_mask=text_input.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        txt_features = text_output.last_hidden_state[:, 0, :].float()
        rewards = model.mlp(txt_features).squeeze(-1)
        rewards = (rewards - model.mean) / model.std
        return rewards.detach().cpu()


# ---------------------------------------------------------------------------
# HPSv2
# ---------------------------------------------------------------------------
class HPSv2Scorer:
    """Batched HPSv2 scoring.

    Bypasses ``hpsv2.score`` (which only accepts one prompt at a time AND
    reloads the checkpoint on every call) by directly grabbing the
    initialised open_clip ViT-H + checkpoint state and running batched
    cross-modal cosine similarity ourselves.
    """

    name = "hpsv2"

    def __init__(self, device=None):
        self.device = _device(device)
        self._model = None
        self._tokenizer = None
        self._preprocess = None

    @staticmethod
    def _patch_hpsv2_vocab() -> None:
        """The hpsv2 PyPI package forgets to ship its bundled
        ``bpe_simple_vocab_16e6.txt.gz``. Copy the identical file from open_clip
        next to hpsv2's vendored open_clip if it's missing."""
        import shutil

        import hpsv2
        import open_clip

        src = Path(open_clip.__file__).parent / "bpe_simple_vocab_16e6.txt.gz"
        dst = (
            Path(hpsv2.__file__).parent / "src" / "open_clip" / "bpe_simple_vocab_16e6.txt.gz"
        )
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)

    def _load(self) -> None:
        if self._model is not None:
            return
        self._patch_hpsv2_vocab()
        import huggingface_hub
        from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
        from hpsv2.utils import hps_version_map

        model, _, preprocess = create_model_and_transforms(
            "ViT-H-14", "laion2B-s32B-b79K", precision="amp", device=str(self.device),
            jit=False, output_dict=True,
        )
        ckpt = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map["v2.0"])
        sd = torch.load(ckpt, map_location=str(self.device), weights_only=False)
        model.load_state_dict(sd["state_dict"])
        model = model.to(self.device).eval()
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = get_tokenizer("ViT-H-14")

    @torch.no_grad()
    def __call__(self, images: Sequence[Image.Image], prompts: Sequence[str]):
        self._load()
        img_batch = torch.stack([self._preprocess(img) for img in images]).to(self.device)
        text_batch = self._tokenizer(list(prompts)).to(self.device)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            out = self._model(img_batch, text_batch)
        # Diagonal of pairwise logits = each image's similarity with ITS OWN prompt.
        scores = torch.diag(out["image_features"] @ out["text_features"].T).float()
        return scores.detach().cpu()


# ---------------------------------------------------------------------------
# Jina CLIP v2 (image-text similarity)
# ---------------------------------------------------------------------------
class JinaClipScorer:
    name = "clip_jina"

    def __init__(self, device=None, dtype=torch.bfloat16):
        self.device = _device(device)
        self.dtype = dtype
        self._model = None
        self._tokenizer = None
        self._image_processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

        self._model = AutoModel.from_pretrained(
            "jinaai/jina-clip-v2",
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device).eval()
        self._tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-clip-v2")
        self._image_processor = AutoImageProcessor.from_pretrained(
            "jinaai/jina-clip-v2", trust_remote_code=True
        )

    @torch.no_grad()
    def __call__(self, images: Sequence[Image.Image], prompts: Sequence[str]):
        self._load()
        img_in = self._image_processor(list(images), return_tensors="pt").to(self.device)
        txt_in = self._tokenizer(
            list(prompts),
            truncation=True,
            padding="longest",
            max_length=1024,
            return_tensors="pt",
        ).to(self.device)
        with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"):
            out = self._model(**img_in, **txt_in)
        return (torch.diag(out.logits_per_image).float() / 100.0).detach().cpu()


# ---------------------------------------------------------------------------
# OpenAI CLIP-L (image-text similarity)
# ---------------------------------------------------------------------------
class OpenAIClipScorer:
    name = "clip_openai"

    def __init__(self, device=None):
        self.device = _device(device)
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import CLIPModel, CLIPProcessor

        self._processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self._model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        ).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, images: Sequence[Image.Image], prompts: Sequence[str]):
        self._load()
        inputs = self._processor(
            text=list(prompts),
            images=list(images),
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            out = self._model(**inputs)
        return (torch.diag(out.logits_per_image).float() / 100.0).detach().cpu()


SCORERS = {
    "aesthetic": AestheticScorer,
    "pick_score": PickScorer,
    "image_reward": ImageRewardScorer,
    "hpsv2": HPSv2Scorer,
    "clip_jina": JinaClipScorer,
    "clip_openai": OpenAIClipScorer,
}
