"""Score a directory of generated images with multiple reward models.

Expected input layout (same as miro/scripts/generate_geneval.py output):

    image_dir/
    ├── 00000/
    │   ├── metadata.jsonl          # one JSON line with at least {"prompt": "..."}
    │   └── samples/
    │       ├── 0000.png
    │       ├── 0001.png
    │       └── ...
    └── 00001/
        ...

Output: a ``rewards.jsonl`` next to results.jsonl (one line per image) with:

    {"filename": "...", "prompt": "...", "aesthetic": 5.31, "pick_score": ..., ...}

Usage (from the miro/eval/rewards directory):

    uv run miro-rewards-score <image_dir> [--scorers aesthetic,pick_score,image_reward,hpsv2,clip_jina,clip_openai] \
        [--batch-size 16] [--outfile <path>]

Run ``uv run miro-rewards-download`` once before the first scoring run to fetch
the Schuhmann aesthetic-predictor checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from tqdm import tqdm

from reward_models import SCORERS


def _load_prompt_dirs(image_dir: Path) -> list[tuple[Path, str, list[Path]]]:
    """Discover (prompt_dir, prompt, [sample_paths]) triples."""
    entries: list[tuple[Path, str, list[Path]]] = []
    for prompt_dir in sorted(image_dir.iterdir()):
        if not prompt_dir.is_dir():
            continue
        meta = prompt_dir / "metadata.jsonl"
        samples_dir = prompt_dir / "samples"
        if not meta.exists() or not samples_dir.is_dir():
            continue
        with open(meta) as f:
            first_line = f.readline().strip()
        if not first_line:
            continue
        prompt = json.loads(first_line)["prompt"]
        sample_paths = sorted(p for p in samples_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
        if sample_paths:
            entries.append((prompt_dir, prompt, sample_paths))
    return entries


def _batched(seq: list, batch_size: int) -> Iterable[list]:
    for i in range(0, len(seq), batch_size):
        yield seq[i : i + batch_size]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument(
        "--scorers",
        default="aesthetic,pick_score,image_reward,hpsv2,clip_jina,clip_openai",
        help="Comma-separated subset of: " + ", ".join(SCORERS.keys()),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--outfile",
        type=Path,
        default=None,
        help="Where to write per-image rewards (default: <image_dir>/rewards.jsonl)",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Optional: where to write aggregated mean scores (default: <image_dir>/rewards_summary.json)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    image_dir: Path = args.image_dir.resolve()
    if not image_dir.is_dir():
        sys.exit(f"image_dir does not exist: {image_dir}")

    outfile = args.outfile or (image_dir / "rewards.jsonl")
    summary_path = args.summary or (image_dir / "rewards_summary.json")

    selected = [s.strip() for s in args.scorers.split(",") if s.strip()]
    unknown = [s for s in selected if s not in SCORERS]
    if unknown:
        sys.exit(f"Unknown scorer(s): {unknown}. Available: {list(SCORERS)}")

    print(f"Image dir: {image_dir}")
    print(f"Scorers:   {selected}")
    print(f"Outfile:   {outfile}")
    print(f"Summary:   {summary_path}")

    # Flatten to one row per image, paired with its prompt
    rows: list[dict] = []
    for _prompt_dir, prompt, sample_paths in _load_prompt_dirs(image_dir):
        for path in sample_paths:
            rows.append({"filename": str(path), "prompt": prompt})
    print(f"Total images: {len(rows)}")
    if not rows:
        sys.exit("No images found.")

    # Instantiate scorers
    device = torch.device(args.device)
    scorers = {name: SCORERS[name](device=device) for name in selected}

    # Per-scorer pass over the whole list keeps GPU memory bounded:
    # one heavy model loaded at a time. After each scorer finishes we
    # flush the partial rewards.jsonl so a crash mid-run doesn't lose
    # the earlier scorers' work.
    for scorer_name, scorer in scorers.items():
        print(f"\n=== Scoring with {scorer_name} ===", flush=True)
        for batch in tqdm(list(_batched(rows, args.batch_size)), desc=scorer_name):
            images = [Image.open(r["filename"]).convert("RGB") for r in batch]
            prompts = [r["prompt"] for r in batch]
            scores = scorer(images, prompts)
            for r, s in zip(batch, scores.tolist()):
                r[scorer_name] = float(s)
        # Free the model GPU memory before loading the next scorer
        for attr in list(vars(scorer).keys()):
            if attr.startswith("_") and not attr.startswith("__"):
                setattr(scorer, attr, None)
        torch.cuda.empty_cache()
        # Flush partial results
        with open(outfile, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"  wrote partial results ({scorer_name} done) to {outfile}", flush=True)

    with open(outfile, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(rows)} rows to {outfile}")

    # Aggregate
    keys = [s for s in selected]
    means = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
    means["n_images"] = len(rows)
    with open(summary_path, "w") as f:
        json.dump(means, f, indent=2)
    print(f"Wrote summary to {summary_path}")
    print()
    print("Mean reward scores")
    print("==================")
    for k in keys:
        print(f"  {k:14s} = {means[k]:+.4f}")


if __name__ == "__main__":
    main()
