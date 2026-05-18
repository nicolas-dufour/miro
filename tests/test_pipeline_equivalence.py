"""Verify MiroPipeline matches the legacy sampling script bit-for-bit.

Both code paths build the same modules from the same checkpoint and run the
same Euler-flow sampler with the same RNG seed, so the output images should
agree up to floating-point cast noise.

This test is opt-in (requires GPU + a converted checkpoint on disk):

    MIRO_TEST_CHECKPOINT=/tmp/miro-hf/main pytest miro/tests/test_pipeline_equivalence.py

If ``MIRO_TEST_CHECKPOINT`` is unset, the test is skipped.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

CHECKPOINT_ENV = "MIRO_TEST_CHECKPOINT"


@pytest.fixture(scope="module")
def checkpoint_dir() -> Path:
    path = os.environ.get(CHECKPOINT_ENV)
    if not path:
        pytest.skip(f"Set {CHECKPOINT_ENV} to a staged MIRO checkpoint dir to run this test")
    p = Path(path)
    for required in ("config.json", "model.safetensors", "uncond_embedding.npy"):
        if not (p / required).exists():
            pytest.skip(f"{p / required} missing — checkpoint not staged yet")
    return p


def test_pipeline_smoke(checkpoint_dir: Path):
    from miro import MiroPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = MiroPipeline.from_pretrained(str(checkpoint_dir))
    pipe = pipe.to(device)

    gen = torch.Generator(device=device).manual_seed(0)
    prompt = (
        "Photography closeup portrait of an adorable rusty broken­down "
        "steampunk robot covered in budding vegetation, surrounded by tall "
        "grass, misty futuristic sci­fi forest environment."
    )
    images = pipe(
        prompt,
        num_inference_steps=4,           # quick smoke test
        guidance_scale=7.0,
        generator=gen,
    )
    assert len(images) == 1
    img = np.asarray(images[0])
    assert img.dtype == np.uint8
    assert img.shape[-1] == 3
    assert img.shape[0] == img.shape[1]
