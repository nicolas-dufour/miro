# GenEval Evaluation

Standalone evaluation environment for [GenEval](https://github.com/djghosh13/geneval) — an object-focused benchmark for text-to-image models.

This is a **separate [pixi](https://pixi.sh) environment** from the main miro project because GenEval depends on mmdet/mmcv/open-clip which conflict with or bloat the training environment. Pixi is used (rather than uv) so the **CUDA toolkit** (nvcc + dev headers + libs) is fetched as a conda dependency — `mmcv` then builds from source against it with sm_90 (H100) support, with no system CUDA install or `CUDA_HOME` setup required.

## Setup

One-time install of the pixi env + Mask2Former weights + checkpoint key-rename patch:

```bash
cd miro/eval/geneval
pixi install               # ~5–10 min: downloads CUDA toolkit, builds mmcv from source
pixi run setup-models      # downloads Mask2Former + patches mmdet 2.x → 3.x key names
```

The `setup-models` task is idempotent — re-running it is a no-op if `models/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_original.pth` already exists.

### What `pixi install` actually does

- Fetches CUDA 12.1 toolkit (`cuda-nvcc`, `cuda-cudart-dev`, `cuda-cccl`, `libcublas-dev`, etc.) from the nvidia conda channel into `.pixi/envs/default/`.
- Sets `CUDA_HOME=$CONDA_PREFIX`, `TORCH_CUDA_ARCH_LIST=9.0`, `FORCE_CUDA=1` via `[activation.env]`.
- Installs PyTorch 2.4.0+cu121 and friends from pypi.
- Builds `mmcv==2.1.0` from source (`no-build-isolation` so it sees torch + CUDA in the env). The pre-built mmcv wheels on pypi / openmmlab CDN only target up to sm_89; H100 needs sm_90.

### Why the checkpoint needs patching

The Mask2Former Swin-S weights distributed by OpenMMLab are in the **mmdet 2.x** layout (`panoptic_head.transformer_decoder.layers.{i}.attentions.{0,1}.attn.*`, `.ffns.0.layers.*`). mmdet 3.x's `Mask2FormerTransformerDecoderLayer` exposes `cross_attn.attn.*`, `self_attn.attn.*`, `ffn.layers.*`. In Mask2Former the **cross-attention runs first**, so `attentions[0]` ↔ `cross_attn` and `attentions[1]` ↔ `self_attn` (the opposite of the naive guess). `patch_mask2former_checkpoint.py` does the rename and keeps a `*_original.pth` backup alongside.

## Usage

### Full pipeline from a Miro checkpoint (recommended)

From the diffusion repo root:

```bash
bash miro/scripts/run_geneval.sh path/to/checkpoint.ckpt \
    [--n-samples 4] [--steps 50] [--cfg 7.0]
```

This runs (1) image generation in the main miro venv via `uv`, (2) Mask2Former-based evaluation in the pixi env, (3) `summary_scores.py` to print per-tag accuracy.

Output lives next to the checkpoint:

```
<ckpt_dir>/geneval_images/<ckpt_name>/
├── 00000/, 00001/, ...   one dir per geneval prompt with samples/
├── hparams.json          generation hyperparams
└── results.jsonl         per-image detector verdict
```

### Just evaluation, on an already-generated directory

```bash
cd miro/eval/geneval
pixi run evaluate <image_dir> --outfile <image_dir>/results.jsonl --model-path ./models
pixi run summary <image_dir>/results.jsonl
```

### Custom evaluation options

```bash
pixi run evaluate <image_dir> \
    --outfile <image_dir>/results.jsonl \
    --model-path ./models \
    --options threshold=0.5 counting_threshold=0.9 clip_model=ViT-L-14
```

## Expected image directory format

```
images/
├── 00000/
│   ├── metadata.jsonl
│   └── samples/
│       ├── 0.png
│       ├── 1.png
│       └── ...
├── 00001/
│   ├── metadata.jsonl
│   └── samples/
│       └── ...
```

## Evaluation options

| Option | Default | Description |
|--------|---------|-------------|
| `threshold` | 0.3 | Object detection confidence threshold |
| `counting_threshold` | 0.9 | Higher threshold for counting tasks |
| `max_objects` | 16 | Max objects per class |
| `max_overlap` | 1.0 | NMS IoU threshold (1.0 = no NMS) |
| `position_threshold` | 0.1 | Tolerance for spatial relation checks |
| `clip_model` | ViT-L-14 | CLIP model for color classification |
| `model` | mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco | Object detector model |
| `bgcolor` | #999 | Background color for cropped objects |
| `crop` | 1 | Whether to crop detected objects |
