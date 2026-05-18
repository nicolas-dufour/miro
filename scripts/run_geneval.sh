#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIRO_ROOT="$(dirname "$SCRIPT_DIR")"
DIFFUSION_ROOT="$(dirname "$MIRO_ROOT")"
GENEVAL_DIR="$MIRO_ROOT/eval/geneval"

usage() {
    echo "Usage: run_geneval.sh <checkpoint> [--n-samples N] [--steps N] [--cfg F]"
    echo ""
    echo "Pipeline: generate images (main miro venv via uv) → evaluate with"
    echo "GenEval (pixi env in miro/eval/geneval). Pixi handles its own CUDA"
    echo "toolkit so mmcv builds from source with sm_90 support, no system"
    echo "CUDA install needed."
    exit 1
}

CHECKPOINT="${1:?$(usage)}"
shift

N_SAMPLES=4
STEPS=50
CFG=7.0

while [[ $# -gt 0 ]]; do
    case $1 in
        --n-samples) N_SAMPLES="$2"; shift 2;;
        --steps) STEPS="$2"; shift 2;;
        --cfg) CFG="$2"; shift 2;;
        *) echo "Unknown option: $1"; usage;;
    esac
done

CKPT_DIR="$(dirname "$(realpath "$CHECKPOINT")")"
CKPT_NAME="$(basename "$CHECKPOINT" .ckpt)"
OUTDIR="$CKPT_DIR/geneval_images/$CKPT_NAME"

echo "=== GenEval Pipeline ==="
echo "Checkpoint: $CHECKPOINT"
echo "Output:     $OUTDIR"
echo ""

# Step 1: Generate images (main miro env)
echo "--- Step 1: Generating images ---"
PYTHONPATH="$DIFFUSION_ROOT" uv run --extra train --project "$DIFFUSION_ROOT" \
    python "$SCRIPT_DIR/generate_geneval.py" \
    --checkpoint "$CHECKPOINT" \
    --outdir "$OUTDIR" \
    --n-samples "$N_SAMPLES" \
    --steps "$STEPS" \
    --cfg "$CFG"

# Step 2: Make sure pixi env is ready (one-time setup builds mmcv from source
# against the CUDA toolkit pixi provides; weights are downloaded and the
# mmdet 2.x state_dict keys are renamed to the 3.x layout).
echo ""
echo "--- Step 2: Preparing eval env (pixi) ---"
pixi run --manifest-path "$GENEVAL_DIR/pixi.toml" --no-progress -- \
    bash -c "[ -f models/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_original.pth ] || pixi run --manifest-path '$GENEVAL_DIR/pixi.toml' setup-models" || \
    pixi run --manifest-path "$GENEVAL_DIR/pixi.toml" setup-models

# Step 3: Evaluate (pixi env, includes CUDA + mmcv + mmdet)
echo ""
echo "--- Step 3: Running GenEval evaluation ---"
pixi run --manifest-path "$GENEVAL_DIR/pixi.toml" evaluate \
    "$OUTDIR" \
    --outfile "$OUTDIR/results.jsonl" \
    --model-path "$GENEVAL_DIR/models"

# Step 4: Summarize
echo ""
echo "--- Step 4: Summary ---"
pixi run --manifest-path "$GENEVAL_DIR/pixi.toml" summary "$OUTDIR/results.jsonl"
