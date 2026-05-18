"""Aggregate a ``rewards.jsonl`` produced by score_images.py.

Prints overall mean, plus per-prompt averages (so the same prompt across
multiple seeds gets a single row). Writes a JSON summary next to the input
unless ``--out`` is given.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


SCORER_KEYS_DEFAULT = (
    "aesthetic",
    "pick_score",
    "image_reward",
    "hpsv2",
    "clip_jina",
    "clip_openai",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rewards_jsonl", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = [json.loads(l) for l in open(args.rewards_jsonl) if l.strip()]
    if not rows:
        raise SystemExit("rewards.jsonl is empty")

    scorer_keys = [k for k in SCORER_KEYS_DEFAULT if k in rows[0]]
    extra_keys = [
        k for k in rows[0] if k not in {"filename", "prompt"} and k not in scorer_keys
    ]
    keys = scorer_keys + extra_keys

    print(f"Total images: {len(rows)}")
    print(f"Unique prompts: {len(set(r['prompt'] for r in rows))}")
    print()

    # Per-image (= per-row) global means
    global_means = {k: statistics.fmean(r[k] for r in rows) for k in keys}
    global_stds = {
        k: (statistics.stdev(r[k] for r in rows) if len(rows) > 1 else 0.0) for k in keys
    }
    print("Per-image means")
    print("===============")
    for k in keys:
        print(f"  {k:14s} = {global_means[k]:+.4f}  ± {global_stds[k]:.4f}")

    # Per-prompt mean-of-samples, then mean-across-prompts
    by_prompt: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_prompt[r["prompt"]].append(r)
    per_prompt_means = {
        k: statistics.fmean(
            statistics.fmean(r[k] for r in samples) for samples in by_prompt.values()
        )
        for k in keys
    }
    print()
    print("Per-prompt means (mean over prompts of mean over samples)")
    print("=========================================================")
    for k in keys:
        print(f"  {k:14s} = {per_prompt_means[k]:+.4f}")

    summary = {
        "n_images": len(rows),
        "n_prompts": len(by_prompt),
        "per_image_mean": global_means,
        "per_image_std": global_stds,
        "per_prompt_mean": per_prompt_means,
    }
    out_path = args.out or args.rewards_jsonl.with_name(
        args.rewards_jsonl.stem + "_summary.json"
    )
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote summary to {out_path}")


if __name__ == "__main__":
    main()
