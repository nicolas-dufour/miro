# MIRO

**MultI-Reward cOnditioned pretraining improves T2I quality and efficiency**

![Qualitative samples from MIRO](assets/teaser.jpg)

[![arXiv](https://img.shields.io/badge/arXiv-2510.25897-b31b1b)](https://arxiv.org/abs/2510.25897)
[![ICML 2026](https://img.shields.io/badge/ICML-2026-blue)](https://icml.cc/)
[![PyPI](https://img.shields.io/badge/pypi-miro--t2i-3776ab)](https://pypi.org/project/miro-t2i/)
[![HF Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hub-nicolas--dufour%2Fmiro-orange)](https://huggingface.co/nicolas-dufour/miro)
[![HF Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Spaces-Demo-yellow)](https://huggingface.co/spaces/nicolas-dufour/miro)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Nicolas Dufour, Lucas Degeorge, Arijit Ghosh, Vicky Kalogeiton, David Picard. _MIRO: MultI-Reward cOnditioned pretraining improves T2I quality and efficiency_. **ICML 2026**.

MIRO is a text-to-image diffusion model that **integrates seven reward signals directly into pretraining** (CLIP, aesthetic, ImageReward, PickScore, HPSv2, VQA, SciScore) rather than aligning with reward models post-hoc. Conditioning a flow-matching RIN backbone on a vector of reward scores during training lets the model map _desired reward levels_ to visual characteristics, and prevents reward hacking by optimising every objective simultaneously. The result: **19× faster convergence**, **GenEval 75 with a 350M model** — competitive with FLUX-dev at **370× less inference compute**.

- 📄 Paper: <https://arxiv.org/abs/2510.25897>
- 🌐 Project page: <https://nicolas-dufour.github.io/miro/>
- 🤗 Models: <https://huggingface.co/nicolas-dufour/miro> (main) · <https://huggingface.co/nicolas-dufour/miro-ablations> (ablations)
- 🐍 PyPI: `pip install miro-t2i`

## Contents

- [Install](#install)
- [Quickstart](#quickstart)
- [Available models](#available-models)
- [Reward conditioning](#reward-conditioning)
- [Specialist baselines](#specialist-baselines)
- [Dataset preprocessing](#dataset-preprocessing)
- [Text-prompt testbed](#text-prompt-testbed)
- [Training](#training)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [License](#license)

## Install

```bash
pip install miro-t2i           # inference only
# or, for training / preprocessing as well:
uv sync --extra train          # from a clone of this repo
```

The PyPI package ships only the inference path. Training, dataset preprocessing, and evaluation scripts live in the GitHub repo and require the `train` extra.

## Quickstart

```python
import torch
from miro import MiroPipeline

pipe = MiroPipeline.from_pretrained("nicolas-dufour/miro")
pipe = pipe.to("cuda", torch.float16)

prompt = (
    "Photography closeup portrait of an adorable rusty broken­down steampunk "
    "robot covered in budding vegetation, surrounded by tall grass, misty "
    "futuristic sci­fi forest environment."
)
images = pipe(
    prompt,
    num_inference_steps=50,
    guidance_scale=7.0,
    num_images_per_prompt=4,
    generator=torch.Generator("cuda").manual_seed(0),
)
for i, img in enumerate(images):
    img.save(f"out_{i}.png")
```

Loading an ablation:

```python
pipe = MiroPipeline.from_pretrained("nicolas-dufour/miro-ablations", variant="miro-no-clip")
```

(See [Available models](#available-models) for the full list of `variant` names.)

### Gradio demo

```bash
pip install miro-t2i[demo]
python -m miro.app
```

The demo spins up a local Gradio UI with the prompt box, sampling controls, seven *positive* and seven *negative* reward sliders (each snapped to `1/64`, matching the training quantisation), 22 example prompts from the testbed, and a `gr.Gallery` output that lets you right-click each generated sample individually plus a download button for the composed 2 × 2 PNG.

The `MIRO_REPO` env var picks the checkpoint (default `nicolas-dufour/miro`); set it to e.g. `nicolas-dufour/miro-ablations` and edit the variant in `miro/app.py` to demo a specialist.

### Deploy on HuggingFace Spaces (ZeroGPU)

The same `app.py` runs unchanged on a HuggingFace **ZeroGPU** Space (free for Pro accounts; GPU is allocated on-demand per inference call). A ready-to-push template lives under [`miro/space/`](miro/space/):

```
space/
├── app.py            # thin shim: ``runpy.run_module("miro.app", run_name="__main__")``
├── requirements.txt  # ``miro-t2i[demo]`` + ``spaces`` + ``torch``
└── README.md         # YAML frontmatter declaring SDK=gradio, hardware=zero-h200
```

To deploy:

```bash
# 1) Create a new Space at huggingface.co/new-space:
#    - SDK: Gradio
#    - Hardware: ZeroGPU (H200)
#    - Repo: nicolas-dufour/miro-demo  (or whatever name)

# 2) Push the template files:
git clone https://huggingface.co/spaces/nicolas-dufour/miro-demo
cp miro/space/* miro-demo/
cd miro-demo
git add . && git commit -m "Initial MIRO demo" && git push
```

The Space landing page (`README.md`) carries the Spaces YAML frontmatter (title, emoji, SDK version, hardware target). On first request HF spins up a ZeroGPU slot, runs the `@spaces.GPU(duration=60)`-decorated `generate()` once, and releases the GPU back to the pool.

If `spaces` isn't installed (i.e., running locally), `app.py` falls through to a no-op decorator and uses the local CUDA device directly.

## Available models

The full 350M-parameter MIRO checkpoint — trained jointly on all seven rewards _and_ on a 50/50 mix of original and synthetic captions — lives at **`nicolas-dufour/miro`**. The fifteen variants (8 ablations + 7 single-reward specialists used as paper baselines) live as subfolders inside **`nicolas-dufour/miro-ablations`** and are loaded with the `variant=` argument.

| Variant | What changed | How to load |
|---|---|---|
| Main MIRO | All 7 rewards + synthetic captions | `MiroPipeline.from_pretrained("nicolas-dufour/miro")` |
| `miro-no-synthetic-captions` | Same recipe but original captions only | `..., variant="miro-no-synthetic-captions"` |
| `miro-no-clip` | Without CLIP reward | `..., variant="miro-no-clip"` |
| `miro-no-aesthetic` | Without aesthetic-quality reward | `..., variant="miro-no-aesthetic"` |
| `miro-no-hpsv2` | Without HPSv2 (human preference) | `..., variant="miro-no-hpsv2"` |
| `miro-no-image-reward` | Without ImageReward | `..., variant="miro-no-image-reward"` |
| `miro-no-pickscore` | Without PickScore | `..., variant="miro-no-pickscore"` |
| `miro-no-sciscore` | Without SciScore | `..., variant="miro-no-sciscore"` |
| `miro-no-vqa` | Without VQAScore | `..., variant="miro-no-vqa"` |
| `miro-only-aesthetic` | Paper baseline: trained on **only** the aesthetic reward | `..., variant="miro-only-aesthetic"` |
| `miro-only-clip` | Paper baseline: trained on **only** CLIP alignment | `..., variant="miro-only-clip"` |
| `miro-only-hpsv2` | Paper baseline: trained on **only** HPSv2 | `..., variant="miro-only-hpsv2"` |
| `miro-only-image-reward` | Paper baseline: trained on **only** ImageReward | `..., variant="miro-only-image-reward"` |
| `miro-only-pickscore` | Paper baseline: trained on **only** PickScore | `..., variant="miro-only-pickscore"` |
| `miro-only-sciscore` | Paper baseline: trained on **only** SciScore | `..., variant="miro-only-sciscore"` |
| `miro-only-vqa` | Paper baseline: trained on **only** VQAScore | `..., variant="miro-only-vqa"` |

`...` is short for `MiroPipeline.from_pretrained("nicolas-dufour/miro-ablations"`.

Every checkpoint shares the same architecture: a RIN flow-matching backbone (4 blocks × 4 processing layers, 256 latents × 1024 dim) trained on the SDXL VAE latent space at 256×256 with FLAN-T5-XL text conditioning.

## Reward conditioning

MIRO's flow model takes a vector of seven reward targets in addition to the text prompt, letting you steer generation at inference time without retraining.

```python
images = pipe(
    prompt,                         # the rusty-robot prompt from above
    reward_targets={
        "clip_score": 1.0,          # strict prompt alignment
        "aesthetic_score": 0.3,     # de-prioritise prettiness
        "image_reward_score": 1.0,  # prioritise general human preference
        # any reward not specified defaults to 1.0
    },
    negative_reward_targets={
        # by default, all zeros; passed through the unconditional branch
        # during classifier-free guidance
    },
    guidance_scale=7.0,
)
```

The seven reward keys are: `clip_score`, `aesthetic_score`, `image_reward_score`, `pick_a_score_score`, `hpsv2_score`, `vqa_score`, `sciscore_score`. All values are normalised to roughly `[0, 1]`; `1.0` requests the strongest version of that signal, `0.0` requests the weakest.

## Specialist baselines

The seven `miro-only-*` variants are the **paper's single-reward specialists** — same architecture and training data as MIRO, but each trained on **only one** reward signal. They're the controls the paper compares MIRO against, and they make the reward-trade-off story very visible:

```python
from miro import MiroPipeline
import torch

clip_only = MiroPipeline.from_pretrained("nicolas-dufour/miro-ablations",
                                          variant="miro-only-clip").to("cuda", torch.float16)
print(clip_only.coherence_keys)
# ('clip_score',)

# Single-reward specialists only know about the one axis they were trained on:
# reward_targets validation enforces that, so unknown keys raise ValueError.
images = clip_only(
    prompt,
    reward_targets={"clip_score": 1.0},   # the only valid key for this checkpoint
    num_inference_steps=50,
    guidance_scale=7.0,
)
```

Each `MiroPipeline` instance exposes `pipe.coherence_keys` — the exact set of reward axes the loaded checkpoint was trained on. For the seven specialists this is a 1-tuple; for the eight `miro-no-*` ablations a 6-tuple (the dropped reward absent); for the main checkpoint the full 7-tuple.

Comparing specialists side-by-side is a one-loop exercise:

```python
prompt = "a photograph of a futuristic temple at dawn"
gen = torch.Generator("cuda").manual_seed(0)
results = {}
for variant in ["miro-only-clip", "miro-only-aesthetic", "miro-only-image-reward",
                 "miro-only-pickscore", "miro-only-hpsv2", "miro-only-vqa", "miro-only-sciscore"]:
    pipe = MiroPipeline.from_pretrained("nicolas-dufour/miro-ablations", variant=variant)
    pipe = pipe.to("cuda", torch.float16)
    results[variant] = pipe(prompt, num_inference_steps=50, guidance_scale=7.0, generator=gen)[0]
```

## Dataset preprocessing

MIRO trains on a mix of two public image-caption datasets:

- [`pixparse/cc12m-wds`](https://huggingface.co/datasets/pixparse/cc12m-wds) — Conceptual 12M, already shipped as webdataset shards.
- [`laion/aesthetics_v2_4.5`](https://huggingface.co/datasets/laion/aesthetics_v2_4.5) — but only the **`aesthetic_score >= 6.0`** subset (filtering on the LAION aesthetic predictor score; we drop everything below 6.0 because the full 4.5+ split is ~10× larger and the lower-aesthetic samples add little to MIRO's per-reward conditioning).

The preprocessing pipeline turns those raw shards into enriched `webdataset` tars that, for every sample, contain:

- the original `.jpg` and `.txt` caption,
- precomputed **SDXL VAE latents** (mean and std, at 256 and 512),
- precomputed **FLAN-T5-XL text embeddings** for the original and synthetic captions,
- **per-sample reward scores** (CLIP, aesthetic, ImageReward, PickScore, HPSv2, VQA, SciScore) for both captions, stored in the sample JSON.

The conversion is split into three GPU-heavy stages so that reward models don't have to share VRAM. Each stage runs against either a single shard, a list of shards, or a contiguous range — pick whichever fits your SLURM array layout.

Starting from raw `cc12m`-style tars (one `.jpg`, one `.txt`, one `.json` per sample):

```bash
# Stage 1 — six reward scores (CLIP, aesthetic, ImageReward, PickScore, HPSv2, SciScore)
# for the original and synthetic captions. Writes one CSV per shard.
uv run --extra train python miro/data/preprocess_data.py \
    --stage rewards \
    --src   /path/to/raw_tars \
    --csv_dir /path/to/scores \
    --shard_range 0-31 \
    [--synthetic_captions captions.tsv]

# Stage 2 — VQAScore. Updates the same CSV; runs in its own process so it does
# not have to coexist with the stage-1 reward models in VRAM.
uv run --extra train python miro/data/preprocess_data.py \
    --stage vqa \
    --src   /path/to/raw_tars \
    --csv_dir /path/to/scores \
    --shard_range 0-31

# Stage 3 — SDXL VAE latents (at 256 and 512) plus FLAN-T5-XL text embeddings
# for the original and synthetic captions. Reads the per-shard CSV, merges the
# rewards into the sample JSON, and writes a brand-new enriched tar to --dest.
uv run --extra train python miro/data/preprocess_data.py \
    --stage vae \
    --src   /path/to/raw_tars \
    --csv_dir /path/to/scores \
    --dest  /path/to/enriched_tars \
    --shard_range 0-31
```

The `--shard_id`, `--shard_ids`, and `--shard_range` flags all map naturally onto a SLURM job array. To regenerate everything in one process (no sharding, no SLURM), pass `--stage all` and omit the shard flags.

### Synthetic captions

The `synth_synth_p50` variant is trained on a 50/50 mix of original and synthetic captions. Provide synthetic captions in either of two ways:

1. Pre-populate `synthetic_caption` and `short_synthetic_caption` in each sample's JSON before stage 1.
2. Pass a TSV file via `--synthetic_captions captions.tsv` with columns `file_name TAB short_caption TAB long_caption` (no header).

### Output layout

Each enriched sample contains:

```
{key}.jpg                         # original image bytes
{key}.txt                         # original caption
{key}.json                        # original metadata + per-reward scores + synthetic captions
{key}.vae_embeddings_mean_256.npy # (4, 32, 32) float16
{key}.vae_embeddings_std_256.npy
{key}.vae_embeddings_mean_512.npy # (4, 64, 64) float16  (only if --no_vae_512 not set)
{key}.vae_embeddings_std_512.npy
{key}.flan_t5_xl_embeddings.npy   # (n_tokens, 2048) float16, trimmed to attention mask
{key}.synthetic_flan_t5_xl_embeddings.npy
```

### Reward-model dependencies

The reward backbones not on PyPI (HPSv2 wrapper, aesthetic predictor, ImageReward, VQAScore) are vendored under [`miro/utils/rewards/`](miro/utils/rewards) — no `cad/` or other external package is required. The preprocessing script will fetch the following from HuggingFace on first run:

- `jinaai/jina-clip-v2` (CLIP score)
- `yuvalkirstain/PickScore_v1` + `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` (PickScore)
- `Jialuo21/SciScore` (SciScore)
- `google/flan-t5-xl` (text embeddings)
- `stabilityai/sdxl-vae` (VAE encoding)
- `ImageReward-v1.0` (downloaded by `miro.utils.rewards.image_reward`)

A single A100-80GB fits stage 1 if HPSv2 is run on its own; otherwise plan for two A100s in stage 1, one in stage 2, and one in stage 3. Stage 3 is disk-bound: budget roughly `1.6×` the size of the raw shards for the enriched output.

## Text-prompt testbed

`miro/datasets/text_prompt_testbed/` holds the **80 fixed prompts** used by the training-time image-logging callback (`miro/callbacks/log_images.py` → `TextCondPromptBed`) to produce a consistent qualitative grid across runs and across ablations. Inference users don't need this directory — it's only consumed during training and by `miro/scripts/sample_*.py`.

Files:

```
prompts.txt                          # one prompt per line
metadata.csv                         # (file_name, text) — pairs each prompt with its embedding filename
flan_t5_xl_embeddings/{name}.npy     # per-prompt FLAN-T5-XL embedding, (n_tokens, 2048) float32
```

The `.npy` files are **not committed** — precompute them locally:

```bash
uv run --extra train python miro/datasets/text_prompt_testbed/precompute_logging_embeddings.py
```

This downloads `google/flan-t5-xl` (~10 GB GPU memory) and writes one `.npy` per row of `metadata.csv` into `flan_t5_xl_embeddings/`, trimmed to the real attention-mask length.

To customise the prompt set, edit `prompts.txt` and `metadata.csv` in lockstep (each row of the CSV pairs a `file_name` stem with its prompt text), then re-run the precompute script. See [`miro/datasets/text_prompt_testbed/README.md`](miro/datasets/text_prompt_testbed/README.md) for the full per-file reference.

The testbed is consumed only by the training-time image-logging callback — neither inference nor evaluation (GenEval and the reward scorers both encode prompts with T5 on the fly) need it.

## Training

```bash
git clone https://github.com/nicolas-dufour/miro
cd miro
uv sync --extra train

# Configure your site (SLURM partition, data root, wandb, …)
cp .env.example .env
$EDITOR .env                              # fill in MIRO_DATA_DIR, MIRO_SLURM_PARTITION, …
set -a; source .env; set +a               # or use direnv
```

Every site-specific value (data root, SLURM partition, wandb entity / project, …) is read from environment variables — see `.env.example` for the full list. `MIRO_DATA_DIR` and `MIRO_SLURM_PARTITION` are required when launching training; everything else has a sensible default.

The training entrypoint is `miro/train.py` (Hydra + Lightning). The main MIRO recipe is `multi_cad_synth` (multi-reward conditioning + 50/50 synthetic captions); each single-reward ablation has its own experiment file under `miro/configs/experiment/`. Checkpoints land under `${root_dir}/miro/checkpoints/${experiment_name}/`; the produced `last.ckpt` is what `miro/scripts/convert_and_upload_to_hf.py` consumes.

### Default: 16× H100 (2 nodes × 8 GPUs)

This is the reference setup used to train the released checkpoint. Effective global batch size 1024, `16-mixed` precision, ~2.65 train it/s, **~52 hours** of wall-clock for 500 k steps.

```bash
# Hand-rolled SLURM launcher (Kyutai partition; adapt to your cluster)
cd miro/slurm && python launch_multicad_synth_16gpu.py --launch
```

### 8× H100 fallback (1 node × 8 GPUs, gradient accumulation)

If you only have a single node available, gradient accumulation preserves the global batch size at 1024. Measured throughput: **≈ 1.45 train it/s** on 8× H100 (vs. 2.65 on 16× H100), so a full 500 k-step training run takes **~96 hours (~4 days)**.

```bash
cd miro/slurm && python launch_multicad_synth_8gpu.py --launch
# Or as a one-shot Hydra invocation:
uv run --extra train python miro/train.py \
    experiment=multi_cad_synth \
    computer.devices=8 computer.num_nodes=1 computer.precision=16-mixed \
    +trainer.accumulate_grad_batches=2 \
    trainer.strategy.static_graph=false \
    trainer.strategy.find_unused_parameters=true \
    data_dir=/path/to/enriched_tars \
    experiment_name=miro_synth_8gpu
```

The two `trainer.strategy.*` overrides are required: the base config sets `static_graph=true`, which trips a DDP assertion under gradient accumulation because the loss applies self-conditioning on only 90 % of steps. The launcher script bakes these in for you. Both launchers otherwise share the same Hydra config — only the GPU count and accumulation factor differ.

## Evaluation

The paper reports two families of metrics: the **seven training-reward scorers** (CLIP, aesthetic, ImageReward, PickScore, HPSv2, VQA, SciScore) and the **GenEval compositional benchmark** (object counts, colours, positions, attributes).

Each evaluator lives in its **own isolated environment**. This is deliberate — these benchmarks are notoriously fragile to set up and their dependency trees actively conflict with each other and with the training environment:

- **GenEval** needs `mmdet 2.x / mmcv 2.1` against a specific CUDA toolkit; the published wheels target up to sm_89, so on H100 (sm_90) we have to rebuild `mmcv` from source. We use [pixi](https://pixi.sh) so the CUDA toolkit comes from conda — no system CUDA install or `CUDA_HOME` setup. The Mask2Former Swin-S checkpoint also needs a key-rename patch (mmdet 2.x → 3.x layout).
- **Reward scorers** need `transformers<4.43` (PickScore + Jina-CLIP broke on the newer `CLIPModel.get_image_features` signature), plus the OpenAI `clip` package (not `open_clip_torch`) used by the aesthetic predictor. We use a separate uv project here too — it's pure PyTorch so no system CUDA needed.
- **Training/inference** pins newer transformers + diffusers and would conflict with both of the above.

Keeping them apart means each one is reproducible in isolation; the cost is one extra `pixi install` / `uv sync` per evaluator.

### Score a single checkpoint on everything

Both evaluators read a flat directory of generated images (one folder per prompt, samples inside). Generate the images once, then score them in each env:

```bash
# 1) Generate images for the GenEval benchmark prompts (553 prompts × N samples)
#    Uses the main miro environment (uv) and writes to <ckpt_dir>/geneval_images/<ckpt_name>/.
bash miro/scripts/run_geneval.sh path/to/last.ckpt --n-samples 4 --steps 50 --cfg 7.0
# `run_geneval.sh` does the full GenEval pipeline: generation in the miro venv,
# Mask2Former evaluation in the pixi env, then a summary print. Image dir lands
# at `<ckpt_dir>/geneval_images/<ckpt_name>/`.

# 2) Score the same images on the seven reward models (separate uv env)
cd miro/eval/rewards
uv sync                              # one-time
uv run miro-rewards-download         # one-time: aesthetic-predictor MLP
uv run miro-rewards-score <ckpt_dir>/geneval_images/<ckpt_name> \
    --outfile <ckpt_dir>/geneval_images/<ckpt_name>/rewards.jsonl
uv run miro-rewards-summary <ckpt_dir>/geneval_images/<ckpt_name>/rewards.jsonl
```

After step 2 you get two artefacts next to the checkpoint:
- `geneval_images/<ckpt_name>/results.jsonl` — per-image Mask2Former verdicts (used for the GenEval score)
- `geneval_images/<ckpt_name>/rewards.jsonl` + `rewards_summary.json` — per-image reward scores plus aggregate means/stds

### Just GenEval, just rewards, or just generation

```bash
# Generate without scoring (skip mmcv/pixi setup)
uv run python miro/scripts/generate_geneval.py \
    --checkpoint path/to/last.ckpt --outdir my_images \
    --n-samples 4 --steps 50 --cfg 7.0

# Score existing images on GenEval only (assumes pixi env + Mask2Former weights already set up)
cd miro/eval/geneval
pixi install                # one-time, ~5–10 min — builds mmcv from source
pixi run setup-models       # one-time: downloads + patches Mask2Former
pixi run evaluate my_images --outfile my_images/results.jsonl --model-path ./models
pixi run summary my_images/results.jsonl

# Score existing images on rewards only
cd miro/eval/rewards && uv sync && uv run miro-rewards-download
uv run miro-rewards-score my_images --outfile my_images/rewards.jsonl
```

See `miro/eval/geneval/README.md` and `miro/eval/rewards/README.md` for the full per-evaluator reference (custom thresholds, scorer selection, output schemas).

### Sampling tools

- `miro/scripts/generate_geneval.py` — generates the GenEval prompt sweep from a `.ckpt`.

## Citation

```bibtex
@inproceedings{dufour2026miro,
  title     = {{MIRO}: {M}ult{I}-{R}eward c{O}nditioned pretraining improves {T2I} quality and efficiency},
  author    = {Dufour, Nicolas and Degeorge, Lucas and Ghosh, Arijit and Kalogeiton, Vicky and Picard, David},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE). The released model weights are likewise distributed under MIT. We do not redistribute the SDXL VAE or the FLAN-T5-XL encoder; the pipeline downloads them on demand from `stabilityai/sdxl-vae` and `google/flan-t5-xl`, which are subject to their respective licenses.