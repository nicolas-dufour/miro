"""Resolve and load static assets shipped inside the ``miro`` package.

The package ships two small precomputed FLAN-T5-XL embeddings used by training
and inference:

- ``flan_t5_xl_uncond.npy`` — unconditional (empty-string) embedding for CFG.
- ``flan_t5_xl_random.npy`` — random-prompt embedding bank, used by the
  ``LogGeneratedImages`` callback to sample negative prompts during training.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


def asset_path(name: str) -> Path:
    """Return the absolute path to an asset shipped with the miro package."""
    return ASSETS_DIR / name


def load_uncond_embedding(file: str | None = None) -> np.ndarray:
    """Load ``flan_t5_xl_uncond.npy`` (the default) or an arbitrary asset.

    Signature is compatible with ``numpy.load`` so this can be used as a Hydra
    ``_target_`` drop-in for the legacy ``cad/utils/flan_t5_xl_uncond.npy``
    config entry.
    """
    if file is None:
        file = str(asset_path("flan_t5_xl_uncond.npy"))
    return np.load(file)


def load_random_prompt_bank(text_embedding_name: str = "flan_t5_xl") -> np.ndarray:
    """Load the random-prompt embedding bank for ``LogGeneratedImages``."""
    return np.load(asset_path(f"{text_embedding_name}_random.npy"))
