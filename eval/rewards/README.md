# Reward-Model Scoring

Standalone reward-model scoring for Miro-generated images. Runs six standard
text-to-image reward models over a directory of generated samples:

| Scorer | Source | Notes |
|---|---|---|
| `aesthetic` | Schuhmann sac+logos+ava1-l14 MLP on OpenAI CLIP ViT-L/14 | predicts a single aesthetic rating (~1-10 scale) |
| `pick_score` | yuvalkirstain/PickScore_v1 | image-text logits / 100 |
| `image_reward` | THUDM ImageReward-v1.0 | trained on human prefs |
| `hpsv2` | xswu/HPSv2 (ViT-H) | human-preference score |
| `clip_jina` | jinaai/jina-clip-v2 | image-text cosine, /100 |
| `clip_openai` | openai/clip-vit-large-patch14 | image-text cosine, /100 |

This is a separate **uv** project from both `miro/` (training) and
`miro/eval/geneval/` (mmdet/mmcv-bound object detection). No conda / pixi / system
CUDA is needed — torch 2.5.1+cu124 ships pre-built kernels that run on H100
(sm_90) out of the box, and the reward models are pure PyTorch.

## Setup

```bash
cd miro/eval/rewards
uv sync                            # installs torch, transformers, image-reward, hpsv2, clip, ...
uv run miro-rewards-download       # one-time: downloads aesthetic-predictor MLP checkpoint
```

The other model weights are auto-downloaded on first use (Hugging Face cache).

## Usage

Score a directory in geneval-layout (one prompt per `NNNNN/`, samples in `NNNNN/samples/`):

```bash
cd miro/eval/rewards
uv run miro-rewards-score path/to/generated/images \
    [--scorers aesthetic,pick_score,image_reward,hpsv2,clip_jina,clip_openai] \
    [--batch-size 16] \
    [--outfile path/to/rewards.jsonl]
```

Defaults: all six scorers, `--outfile <image_dir>/rewards.jsonl`,
`--summary <image_dir>/rewards_summary.json`.

Per-scorer pass over all images (one model loaded at a time) keeps GPU RAM
bounded — ImageReward, PickScore, HPSv2 are ~1-2 GB each.

### Summary of an existing rewards.jsonl

```bash
uv run miro-rewards-summary path/to/rewards.jsonl
```

Prints per-image and per-prompt means with stddev.

### Output schema

`rewards.jsonl` — one line per image:

```json
{"filename": "/abs/path/00042/samples/0001.png",
 "prompt": "a photo of a frisbee",
 "aesthetic": 5.31, "pick_score": 0.22, "image_reward": 0.81,
 "hpsv2": 0.28, "clip_jina": 0.27, "clip_openai": 0.31}
```

`rewards_summary.json` — global stats:

```json
{"n_images": 2212, "n_prompts": 553,
 "per_image_mean": {...}, "per_image_std": {...}, "per_prompt_mean": {...}}
```

## Notes

- `transformers<4.43` is pinned because newer versions changed
  `CLIPModel.get_image_features` signatures used by PickScore and Jina CLIP.
- The OpenAI `clip` package (not `open_clip_torch`) is required by the
  aesthetic predictor checkpoint — both are installed.
- Reward models are loaded lazily; if you pass `--scorers clip_openai`, only
  that one gets pulled.
