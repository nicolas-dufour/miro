"""Convert legacy MIRO ``.ckpt`` checkpoints into the public HuggingFace layout.

For each checkpoint folder in ``--ckpt-root`` (default
``checkpoints_ground_truth/``):

1. Read ``config.yaml`` and emit a clean ``config.json`` (no Hydra targets).
2. Read ``last.ckpt``, drop optimizer state and non-EMA weights, rename
   ``ema_network.*`` → ``network.*``, and save the resulting state dict as a
   fp16 ``model.safetensors``.
3. Copy the FLAN-T5-XL unconditional embedding (or recompute it) as
   ``uncond_embedding.npy``.
4. Render a model-card ``README.md``.
5. Optionally upload to the HuggingFace Hub (``--push``).

Usage:

    # Stage every checkpoint locally (no network calls):
    uv run python miro/scripts/convert_and_upload_to_hf.py \\
        --ckpt-root miro/checkpoints_ground_truth \\
        --staging-dir /tmp/miro-hf

    # Stage and upload everything:
    uv run python miro/scripts/convert_and_upload_to_hf.py \\
        --ckpt-root miro/checkpoints_ground_truth \\
        --staging-dir /tmp/miro-hf \\
        --push

    # Single checkpoint:
    uv run python miro/scripts/convert_and_upload_to_hf.py \\
        --ckpt-dir miro/checkpoints_ground_truth/CC12M_256_RIN_small_flow_multi_cad \\
        --staging-dir /tmp/miro-hf
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from safetensors.torch import save_file

# Maps every local checkpoint folder name to (repo_id, subfolder).
#
# The full MIRO recipe is the one trained on 50% synthetic captions
# (``synth_synth_p50``), so it lives at the root of ``nicolas-dufour/miro``.
# The pre-synthetic-caption baseline and the seven single-reward ablations
# live as subfolders under ``nicolas-dufour/miro-ablations``.
RELEASES: dict[str, tuple[str, str | None]] = {
    "CC12M_256_RIN_small_flow_multi_cad_synth_synth_p50": ("nicolas-dufour/miro",            None),
    "CC12M_256_RIN_small_flow_multi_cad":                 ("nicolas-dufour/miro-ablations",  "miro-no-synthetic-captions"),
    "CC12M_256_RIN_small_flow_multi_cad_no_aesthetic":    ("nicolas-dufour/miro-ablations",  "miro-no-aesthetic"),
    "CC12M_256_RIN_small_flow_multi_cad_no_clip":         ("nicolas-dufour/miro-ablations",  "miro-no-clip"),
    "CC12M_256_RIN_small_flow_multi_cad_no_hpsv2":        ("nicolas-dufour/miro-ablations",  "miro-no-hpsv2"),
    "CC12M_256_RIN_small_flow_multi_cad_no_image_reward": ("nicolas-dufour/miro-ablations",  "miro-no-image-reward"),
    "CC12M_256_RIN_small_flow_multi_cad_no_pick":         ("nicolas-dufour/miro-ablations",  "miro-no-pickscore"),
    "CC12M_256_RIN_small_flow_multi_cad_no_sciscore":     ("nicolas-dufour/miro-ablations",  "miro-no-sciscore"),
    "CC12M_256_RIN_small_flow_multi_cad_no_vqa":          ("nicolas-dufour/miro-ablations",  "miro-no-vqa"),
    # Single-reward specialists trained on Jean Zay (compared against MIRO
    # in the paper) — variant trained with ONLY one reward signal active.
    "CC12M_256_RIN_small_flow_aesthetic_score_cad":       ("nicolas-dufour/miro-ablations",  "miro-only-aesthetic"),
    "CC12M_256_RIN_small_flow_clip_score_cad":            ("nicolas-dufour/miro-ablations",  "miro-only-clip"),
    "CC12M_256_RIN_small_flow_hpsv2_score_cad":           ("nicolas-dufour/miro-ablations",  "miro-only-hpsv2"),
    "CC12M_256_RIN_small_flow_image_reward_score_cad":    ("nicolas-dufour/miro-ablations",  "miro-only-image-reward"),
    "CC12M_256_RIN_small_flow_pick_a_score_score_cad":    ("nicolas-dufour/miro-ablations",  "miro-only-pickscore"),
    "CC12M_256_RIN_small_flow_sciscore_score_cad":        ("nicolas-dufour/miro-ablations",  "miro-only-sciscore"),
    "CC12M_256_RIN_small_flow_vqa_score_cad":             ("nicolas-dufour/miro-ablations",  "miro-only-vqa"),
}

# Kwargs the public ``CADRINTextCond`` accepts. Used to filter the legacy
# Hydra config (which carries dozens of training-only fields).
ALLOWED_NETWORK_KEYS = {
    "data_size", "data_dim", "num_input_channels", "num_latents",
    "latents_dim", "label_dim", "num_cond_tokens",
    "num_processing_layers", "num_blocks", "patch_size",
    "read_write_heads", "compute_heads",
    "latent_mlp_multiplier", "data_mlp_multiplier",
    "compute_dropout", "rw_stochastic_depth", "compute_stochastic_depth",
    "num_text_registers", "coherence_keys",
    "coherence_dropout", "dropout_strategy",
    "use_cond_rin_block", "concat_cond_token_to_latents",
    "use_self_conditioning",
}

# Human-readable descriptions, keyed by the *subfolder name* in the HF repo
# (matches the second element of every RELEASES value, plus ``None`` for the
# main repo at ``nicolas-dufour/miro``).
ABLATION_DESCRIPTIONS = {
    None: (
        "**Main MIRO checkpoint.** Trained jointly on all seven reward signals "
        "(CLIP, aesthetic, ImageReward, PickScore, HPSv2, VQAScore, SciScore) "
        "with a 50/50 mix of original and synthetic captions."
    ),
    "miro-no-synthetic-captions": (
        "Ablation: same as main MIRO but trained only on original captions "
        "(no synthetic-caption augmentation)."
    ),
    "miro-no-aesthetic":    "Ablation: trained without the aesthetic-quality reward.",
    "miro-no-clip":         "Ablation: trained without the CLIP text-image alignment reward.",
    "miro-no-hpsv2":        "Ablation: trained without the HPSv2 human-preference reward.",
    "miro-no-image-reward": "Ablation: trained without ImageReward.",
    "miro-no-pickscore":    "Ablation: trained without PickScore.",
    "miro-no-sciscore":     "Ablation: trained without SciScore.",
    "miro-no-vqa":          "Ablation: trained without VQAScore.",
    # Single-reward specialists (paper baselines): trained on ONE reward.
    "miro-only-aesthetic":    "Single-reward baseline: trained with **only** the aesthetic reward active.",
    "miro-only-clip":         "Single-reward baseline: trained with **only** the CLIP text-image alignment reward.",
    "miro-only-hpsv2":        "Single-reward baseline: trained with **only** the HPSv2 human-preference reward.",
    "miro-only-image-reward": "Single-reward baseline: trained with **only** ImageReward.",
    "miro-only-pickscore":    "Single-reward baseline: trained with **only** PickScore.",
    "miro-only-sciscore":     "Single-reward baseline: trained with **only** SciScore.",
    "miro-only-vqa":          "Single-reward baseline: trained with **only** VQAScore.",
}


def _resolve_or(value, fallback):
    return fallback if isinstance(value, str) and value.startswith("${") else value


def _flatten(value):
    """Resolve OmegaConf containers to plain Python types."""
    if hasattr(value, "_to_container"):
        return value._to_container(resolve=True)
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def build_clean_config(full_cfg) -> dict:
    """Turn the legacy Hydra config into a plain JSON config the pipeline can load."""
    network_cfg = full_cfg.model.network.instance
    raw = OmegaConf.to_container(network_cfg, resolve=False)
    network_kwargs = {k: v for k, v in raw.items() if k in ALLOWED_NETWORK_KEYS}
    network_kwargs["data_size"] = _resolve_or(
        network_kwargs.get("data_size"), full_cfg.data.data_resolution
    )
    network_kwargs["label_dim"] = _resolve_or(
        network_kwargs.get("label_dim"), full_cfg.data.label_dim
    )
    network_kwargs["num_cond_tokens"] = _resolve_or(
        network_kwargs.get("num_cond_tokens"), full_cfg.data.num_cond_tokens
    )
    network_kwargs["coherence_keys"] = OmegaConf.to_container(
        full_cfg.model.network.coherence_keys, resolve=True
    )
    # Legacy CAD always has the previous-latents residual path enabled.
    network_kwargs.setdefault("use_self_conditioning", True)

    pre = OmegaConf.to_container(full_cfg.model.preconditioning, resolve=True)
    preconditioning_kwargs = {
        "num_latents": pre.get("num_latents") if isinstance(pre.get("num_latents"), int) else network_kwargs["num_latents"],
        "latents_dim": pre.get("latents_dim") if isinstance(pre.get("latents_dim"), int) else network_kwargs["latents_dim"],
        "do_normalization": pre.get("do_normalization", True),
        "sigma_data": pre.get("sigma_data", 0.5),
        "do_gradnorm_reweighting": pre.get("do_gradnorm_reweighting", True),
        "logvar_channels": pre.get("logvar_channels", 128),
        "logvar_mlp_layers": 0,
    }

    img_resolution = int(full_cfg.data.img_resolution)
    data_resolution = int(full_cfg.data.data_resolution)
    channel_wise = bool(full_cfg.model.channel_wise_normalisation)

    data_preprocessing_kwargs = {
        "input_key_mean": f"vae_embeddings_mean_{img_resolution}",
        "input_key_std": f"vae_embeddings_std_{img_resolution}",
        "output_key_root": "x_0",
        "vae_sample": True,
        "channel_wise_normalisation": channel_wise,
        "model_type": "sdxl",
    }

    postprocessing_kwargs = {
        "channel_wise_normalisation": channel_wise,
        "model_type": "sdxl",
    }

    return {
        "network": network_kwargs,
        "preconditioning": preconditioning_kwargs,
        "data_preprocessing": data_preprocessing_kwargs,
        "postprocessing": postprocessing_kwargs,
        "scheduler": {"start": 1, "end": 0, "clip_min": 1e-9},
        "coherence_keys": list(network_kwargs["coherence_keys"]),
        "sampler_defaults": {
            "num_steps": 50,
            "guidance_scale": float(full_cfg.model.cfg_rate),
            "sigma_data": preconditioning_kwargs["sigma_data"],
        },
        "data_resolution": data_resolution,
        "img_resolution": img_resolution,
        "max_text_len": int(full_cfg.data.num_cond_tokens),
        "model_type": "sdxl",
        "vae_repo": "stabilityai/sdxl-vae",
        "text_encoder_repo": "google/flan-t5-xl",
    }


def extract_inference_state(ckpt_state: dict[str, torch.Tensor], dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Pull the inference-only tensors from a full Lightning state_dict.

    Renames ``ema_network.*`` → ``network.*`` so the pipeline sees a single
    namespace, and drops the non-EMA ``network.*`` weights and the SDXL VAE
    parameters (which we reload from the Stability repo at runtime).
    """
    out: dict[str, torch.Tensor] = {}

    def add(prefix_in, prefix_out, *, skip=()):
        p = prefix_in + "."
        for k, v in ckpt_state.items():
            if not k.startswith(p):
                continue
            sub = k[len(p):]
            # Strip ``_orig_mod.`` added by ``torch.compile`` wrapping; some
            # runs (the single-reward specialists) saved compiled state dicts.
            if sub.startswith("_orig_mod."):
                sub = sub[len("_orig_mod."):]
            if any(sub.startswith(s) for s in skip):
                continue
            out[f"{prefix_out}.{sub}"] = v.to(dtype) if v.is_floating_point() else v

    # ``output_scaling`` is a per-VAE-channel learnable scalar that the legacy
    # ablation runs initialised at 1.0 and never trained — it's identity in
    # every checkpoint that carries it. Drop it so the public
    # ``CADRINTextCond`` (which has no such attribute) can strict-load.
    add("ema_network", "network", skip=("output_scaling",))
    add("preconditioning", "preconditioning")
    add("data_preprocessing", "data_preprocessing")
    add("postprocessing", "postprocessing", skip=("vae.",))
    return out


def load_uncond_embedding(repo_root: Path) -> np.ndarray:
    """Return the precomputed FLAN-T5-XL unconditional embedding.

    Pulled from the asset bundled with the miro package (``miro/assets/
    flan_t5_xl_uncond.npy``); falls back to encoding the empty string with
    FLAN-T5-XL on the fly if the asset is missing.
    """
    bundled = repo_root / "assets" / "flan_t5_xl_uncond.npy"
    if bundled.exists():
        return np.load(bundled).astype(np.float32)

    print(f"[warn] {bundled} missing; computing fresh from google/flan-t5-xl", file=sys.stderr)
    from transformers import AutoTokenizer, T5EncoderModel

    tok = AutoTokenizer.from_pretrained("google/flan-t5-xl")
    enc = T5EncoderModel.from_pretrained("google/flan-t5-xl").eval()
    out = tok([""], return_tensors="pt", padding="longest", truncation=True, max_length=77)
    with torch.no_grad():
        emb = enc(**out).last_hidden_state[0].float().cpu().numpy()
    return emb.astype(np.float32)


# Static masonry teaser at ``miro/assets/teaser.jpg`` copied into every
# staged repo so the model card renders the same hero gallery everywhere.
TEASER_FILENAME = "teaser.jpg"
TEASER_SOURCE = Path(__file__).resolve().parents[1] / "assets" / TEASER_FILENAME


PRETTY_NAMES = {
    None: "MIRO (main)",
    "miro-no-synthetic-captions": "MIRO without synthetic captions",
    "miro-no-aesthetic":          "MIRO without the aesthetic reward",
    "miro-no-clip":               "MIRO without the CLIP reward",
    "miro-no-hpsv2":              "MIRO without the HPSv2 reward",
    "miro-no-image-reward":       "MIRO without ImageReward",
    "miro-no-pickscore":          "MIRO without PickScore",
    "miro-no-sciscore":           "MIRO without SciScore",
    "miro-no-vqa":                "MIRO without VQAScore",
    "miro-only-aesthetic":        "Aesthetic-only specialist (paper baseline)",
    "miro-only-clip":             "CLIP-only specialist (paper baseline)",
    "miro-only-hpsv2":            "HPSv2-only specialist (paper baseline)",
    "miro-only-image-reward":     "ImageReward-only specialist (paper baseline)",
    "miro-only-pickscore":        "PickScore-only specialist (paper baseline)",
    "miro-only-sciscore":         "SciScore-only specialist (paper baseline)",
    "miro-only-vqa":              "VQAScore-only specialist (paper baseline)",
}


def render_model_card(name: str, subfolder: str | None, config: dict, n_params: int) -> str:
    description = ABLATION_DESCRIPTIONS.get(subfolder, ABLATION_DESCRIPTIONS[None])
    pretty = PRETTY_NAMES.get(subfolder, name)
    reward_keys = ", ".join(f"`{k}`" for k in config["coherence_keys"])
    if subfolder is None:
        load_call = 'MiroPipeline.from_pretrained("nicolas-dufour/miro")'
    else:
        load_call = (
            f'MiroPipeline.from_pretrained(\n'
            f'    "nicolas-dufour/miro-ablations", variant="{subfolder}",\n'
            f')'
        )

    body = f"""---
license: mit
library_name: miro-t2i
tags:
  - text-to-image
  - diffusion
  - flow-matching
  - miro
  - reward-conditioning
pipeline_tag: text-to-image
---

# {pretty}

![Qualitative samples from MIRO]({TEASER_FILENAME})

<sub>Qualitative samples from the released MIRO checkpoint — same gallery as the
teaser of the [project page](https://nicolas-dufour.github.io/miro/).</sub>

{description}

This checkpoint accompanies the paper
**MIRO: MultI-Reward cOnditioned pretraining improves T2I quality and efficiency**
(Dufour, Degeorge, Ghosh, Kalogeiton, Picard — ICML 2026).

| | |
|---|---|
| **Paper** | <https://arxiv.org/abs/2510.25897> |
| **Project page** | <https://nicolas-dufour.github.io/miro/> |
| **Code** | <https://github.com/nicolas-dufour/miro> |
| **Parameters** | {n_params/1e6:.1f}M |
| **Resolution** | {config['img_resolution']}×{config['img_resolution']} (SDXL VAE latent space) |
| **Architecture** | RIN flow-matching backbone, FLAN-T5-XL text conditioning |
| **Training data** | [CC12M](https://huggingface.co/datasets/pixparse/cc12m-wds) + [LAION Aesthetics v2 4.5](https://huggingface.co/datasets/laion/aesthetics_v2_4.5) (6.0+ aesthetic subset) |
| **Reward signals** | {reward_keys} |
| **Weights** | `model.safetensors`, **fp32** (EMA master weights — ready for finetuning) |

## Install

```bash
pip install miro-t2i
```

`miro-t2i` is the public PyPI package; it imports as `import miro`. The first
call to `MiroPipeline.from_pretrained(...)` will additionally fetch
[`google/flan-t5-xl`](https://huggingface.co/google/flan-t5-xl) (text encoder)
and [`stabilityai/sdxl-vae`](https://huggingface.co/stabilityai/sdxl-vae)
(latent decoder) from the Hub.

## Usage

```python
import torch
from miro import MiroPipeline

pipe = {load_call}
pipe = pipe.to("cuda", torch.float16)

prompt = (
    "Photography closeup portrait of an adorable rusty broken­down steampunk "
    "robot covered in budding vegetation, surrounded by tall grass, misty "
    "futuristic sci­fi forest environment."
)
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.0)[0]
image.save("out.png")
```

### Reward conditioning

MIRO conditions the flow model on a vector of reward targets in addition to the
text prompt. By default every reward is requested at its maximum (`1.0`); you
can override individual axes to bias generation toward a particular trade-off:

```python
image = pipe(
    "a chest x-ray showing pneumonia",
    reward_targets={{
        "clip_score": 1.0,        # strict prompt alignment
        "aesthetic_score": 0.3,   # de-prioritise prettiness
        "sciscore_score": 1.0,    # prioritise scientific accuracy
        # any reward not listed defaults to 1.0
    }},
    negative_reward_targets={{
        # zeros by default; what to push the unconditional branch toward
    }},
    guidance_scale=7.0,
)[0]
```

The seven reward dimensions are:

| Reward | Normalised range | What it measures |
|---|---|---|
| `clip_score` | ~[0, 1] | CLIP text–image alignment |
| `aesthetic_score` | ~[0, 1] | LAION aesthetic-quality predictor |
| `image_reward_score` | ~[0, 1] | ImageReward (general preference model) |
| `pick_a_score_score` | ~[0, 1] | PickScore (human preference) |
| `hpsv2_score` | ~[0, 1] | HPSv2 (human preference v2) |
| `vqa_score` | ~[0, 1] | VQAScore (compositional faithfulness) |
| `sciscore_score` | ~[0, 1] | SciScore (scientific-image plausibility) |

## Reported benchmarks

The paper reports the following headline numbers for the **main MIRO** model
(this repo's `nicolas-dufour/miro`):

| Metric | MIRO (350M) | FLUX-dev (12B) |
|---|---|---|
| GenEval (overall) | **75** (with inference-time reward tuning) / 68 (default) | 67 |
| Inference compute | **1×** | ~370× |
| Aesthetic-metric convergence vs. baseline pretraining | **19×** faster | — |

Per-variant scores (GenEval, FID, individual reward scores) for the eight
ablations are reported in the paper's ablation tables. Please refer to
[arXiv:2510.25897](https://arxiv.org/abs/2510.25897) for the full breakdown.

## Training compute and data

- **Default hardware**: 2 nodes × 8 H100 GPUs (16× H100, `16-mixed` precision)
- **Optimiser**: LAMB, lr 1e-3 (5k warmup → cosine decay), weight decay 1e-2
- **Batch size**: 1024 globally (64 per GPU on 16× H100), gradient-clip 2.0
- **Steps**: 500 k (≈ ~29 epochs over the enriched training set)
- **Wall-clock on 16× H100**: ~52 hours (≈ 2.65 train it/s sustained)
- **8-GPU fallback**: 1 node × 8 H100 with `trainer.accumulate_grad_batches=2`,
  measured at **≈ 1.45 train it/s** → ~96 hours (~4 days) end-to-end.
  Requires `trainer.strategy.static_graph=false` and
  `trainer.strategy.find_unused_parameters=true` to play well with the
  self-conditioning skip in the loss; both flags are set automatically by
  `miro/slurm/launch_multicad_synth_8gpu.py`.
- **Data**: [CC12M](https://huggingface.co/datasets/pixparse/cc12m-wds) +
  [LAION Aesthetics v2 4.5](https://huggingface.co/datasets/laion/aesthetics_v2_4.5)
  filtered to `aesthetic_score >= 6.0` (the higher-quality subset), encoded to
  SDXL VAE latents at 256 resolution. Each sample is paired with seven reward
  scores and FLAN-T5-XL embeddings of both the original and a synthetic
  caption, computed by
  [`miro/data/preprocess_data.py`](https://github.com/nicolas-dufour/miro/blob/main/data/preprocess_data.py).

## Limitations and intended use

This checkpoint is a research artifact released to reproduce and build on the
MIRO paper. Known limitations:

- **Resolution**: 256×256 only. Higher-resolution outputs require upscaling.
- **Domain**: trained on web-scraped image–caption pairs (CC12M + LAION
  Aesthetics 6.0). Inherits the biases of those datasets — including
  under-representation of many cultures, languages, and concepts, and the
  presence of stereotypes. Generations may reflect or amplify these biases.
- **Reward-model biases**: the seven reward predictors used during training
  encode their own biases (e.g. aesthetic and human-preference models reflect
  the taste of their annotator pools). Conditioning on these rewards inherits
  and can sharpen those biases.
- **Not for safety-critical use**: outputs are not factual and the SciScore
  reward does not guarantee scientific accuracy.
- **No safety filter** is shipped with the model; users deploying it in
  user-facing settings should add their own.

The model is released under the MIT license; the SDXL VAE and FLAN-T5-XL
encoder it depends on at inference time are loaded from
[`stabilityai/sdxl-vae`](https://huggingface.co/stabilityai/sdxl-vae) and
[`google/flan-t5-xl`](https://huggingface.co/google/flan-t5-xl) and are
subject to their respective licenses.

## Citation

```bibtex
@inproceedings{{dufour2026miro,
  title     = {{{{MIRO}}: {{M}}ult{{I}}-{{R}}eward c{{O}}nditioned pretraining improves {{T2I}} quality and efficiency}},
  author    = {{Dufour, Nicolas and Degeorge, Lucas and Ghosh, Arijit and Kalogeiton, Vicky and Picard, David}},
  booktitle = {{International Conference on Machine Learning (ICML)}},
  year      = {{2026}}
}}
```

## License

MIT — see <https://github.com/nicolas-dufour/miro/blob/main/LICENSE>.
"""
    return body


def stage_checkpoint(
    ckpt_dir: Path,
    staging_root: Path,
    *,
    dtype: torch.dtype,
    repo_root: Path,
) -> tuple[Path, str, str | None]:
    """Convert a single checkpoint into staging_root/<subfolder>/.

    Returns (staged_dir, repo_id, subfolder).
    """
    name = ckpt_dir.name
    if name not in RELEASES:
        raise KeyError(f"{name!r} is not in RELEASES; cannot decide a repo id")
    repo_id, subfolder = RELEASES[name]

    # Per the user-decided layout, the main repo has files at the root and the
    # ablations repo has each variant in a subfolder.
    if subfolder is None:
        staged_dir = staging_root / "main"
    else:
        staged_dir = staging_root / "ablations" / subfolder
    staged_dir.mkdir(parents=True, exist_ok=True)

    print(f"[+] {name}  →  {repo_id}{('/' + subfolder) if subfolder else ''}")
    print(f"    staging: {staged_dir}")

    cfg_path = ckpt_dir / "config.yaml"
    ckpt_path = ckpt_dir / "last.ckpt"
    if not cfg_path.exists() or not ckpt_path.exists():
        raise FileNotFoundError(f"Missing config.yaml or last.ckpt in {ckpt_dir}")

    with open(cfg_path) as f:
        full_cfg = OmegaConf.create(yaml.safe_load(f))
    config = build_clean_config(full_cfg)

    full = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd_in = full["state_dict"] if "state_dict" in full else full
    sd_out = extract_inference_state(sd_in, dtype=dtype)

    save_file(sd_out, str(staged_dir / "model.safetensors"))
    with open(staged_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    uncond = load_uncond_embedding(repo_root)
    np.save(staged_dir / "uncond_embedding.npy", uncond.astype(np.float32))

    # Ship the masonry teaser image alongside the weights so the model card
    # renders the same hero gallery on every repo.
    if TEASER_SOURCE.exists():
        shutil.copyfile(TEASER_SOURCE, staged_dir / TEASER_FILENAME)
    else:
        print(f"    [warn] teaser image missing at {TEASER_SOURCE}; "
              f"the model card will reference a missing image")

    n_params = sum(
        v.numel() for k, v in sd_out.items() if k.startswith("network.") and v.is_floating_point()
    )
    (staged_dir / "README.md").write_text(
        render_model_card(name, subfolder, config, n_params)
    )

    print(f"    safetensors keys: {len(sd_out)}  (network params: {n_params/1e6:.1f}M)")
    print(f"    config.json fields: {list(config.keys())}")
    return staged_dir, repo_id, subfolder


def upload_staged(staged_dir: Path, repo_id: str, subfolder: str | None, *, private: bool):
    """Push a staged folder to the Hub. Requires HF auth (HF_TOKEN env or `huggingface-cli login`)."""
    from huggingface_hub import HfApi

    api = HfApi()
    # ``private`` is honoured only on initial creation; for an already-existing
    # repo, set visibility separately via ``update_repo_settings``.
    api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model", private=private)
    if private:
        try:
            api.update_repo_settings(repo_id=repo_id, repo_type="model", private=True)
        except Exception as exc:
            print(f"    [warn] update_repo_settings failed: {exc}")
    print(f"    uploading to https://huggingface.co/{repo_id}{('/' + subfolder) if subfolder else ''}")
    api.upload_folder(
        folder_path=str(staged_dir),
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=subfolder or ".",
        commit_message=f"Upload {subfolder or 'main'} weights",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt-root", type=Path, help="Directory containing all checkpoint folders.")
    parser.add_argument("--ckpt-dir", type=Path, help="Single checkpoint folder (overrides --ckpt-root).")
    parser.add_argument("--staging-dir", type=Path, required=True, help="Where to materialise the converted files.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Path to the miro package root; used to find the bundled assets/flan_t5_xl_uncond.npy.")
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--push", action="store_true", help="Upload the staged folders to the Hub.")
    visibility = parser.add_mutually_exclusive_group()
    visibility.add_argument("--private", dest="private", action="store_true", default=True,
                            help="Create / keep the target repos private (default).")
    visibility.add_argument("--public", dest="private", action="store_false",
                            help="Make the target repos public.")
    parser.add_argument("--only", nargs="*", help="Restrict to a list of checkpoint folder names.")
    parser.add_argument("--clean-staging", action="store_true", help="Delete --staging-dir before staging.")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    if args.clean_staging and args.staging_dir.exists():
        shutil.rmtree(args.staging_dir)
    args.staging_dir.mkdir(parents=True, exist_ok=True)

    if args.ckpt_dir:
        ckpts = [args.ckpt_dir]
    elif args.ckpt_root:
        ckpts = sorted(p for p in args.ckpt_root.iterdir() if p.is_dir() and p.name in RELEASES)
    else:
        parser.error("Provide --ckpt-dir or --ckpt-root")

    if args.only:
        keep = set(args.only)
        ckpts = [p for p in ckpts if p.name in keep]

    if not ckpts:
        parser.error("No checkpoints to process")

    staged: list[tuple[Path, str, str | None]] = []
    for ckpt in ckpts:
        staged_dir, repo_id, subfolder = stage_checkpoint(
            ckpt, args.staging_dir, dtype=dtype, repo_root=args.repo_root
        )
        staged.append((staged_dir, repo_id, subfolder))

    if args.push:
        visibility = "private" if args.private else "public"
        print(f"\n[push] visibility: {visibility}")
        for staged_dir, repo_id, subfolder in staged:
            upload_staged(staged_dir, repo_id, subfolder, private=args.private)
        print("\n[done] all uploads complete")
    else:
        print("\n[done] staging complete. Re-run with --push to upload.")


if __name__ == "__main__":
    main()
