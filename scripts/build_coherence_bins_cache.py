"""Pre-compute coherence-bin caches for TextWIDSDataset.

Reads only the per-sample ``.json`` members from a directory of webdataset
tars (or from the WIDS shardlist), substitutes synthetic scores with the
requested probability, and writes the same JSON cache that
``TextWIDSDataset`` looks for under ``<root>/coherence_bins_*.json``.

Doing this offline avoids the slow path where the dataset re-reads every
sample (including all `.npy` embeddings) through `_selective_decode`.

Usage::

    uv run --extra train python miro/scripts/build_coherence_bins_cache.py \\
        --root $MIRO_DATA_DIR/cc12m/train \\
        --synthetic-prob 0.5 \\
        --max-samples 100000 \\
        --num-workers 32
"""
from __future__ import annotations
import argparse
import json
import random
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
from tqdm import tqdm

# Reuse _compute_bins from the dataset module so caches are bit-identical to
# what TextWIDSDataset would produce.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from miro.data.text_wids_dataset import _compute_bins


DEFAULT_SCORES = [
    "clip_score",
    "aesthetic_score",
    "image_reward_score",
    "pick_a_score_score",
    "hpsv2_score",
    "vqa_score",
    "sciscore_score",
]


def _sample_scores_from_tar(args: tuple) -> dict[str, list[float]]:
    tar_path, scores, synthetic_prob, samples_per_tar, seed = args
    rng = random.Random(seed)
    out = {s: [] for s in scores}
    try:
        with tarfile.open(tar_path, "r") as tf:
            # Each sample's json member name is "<key>.json"
            members = [m for m in tf.getmembers() if m.name.endswith(".json")]
            if samples_per_tar is not None and samples_per_tar < len(members):
                members = rng.sample(members, samples_per_tar)
            for m in members:
                try:
                    f = tf.extractfile(m)
                    if f is None:
                        continue
                    meta = json.loads(f.read().decode("utf-8"))
                except Exception:
                    continue
                use_synth = synthetic_prob > 0 and rng.random() < synthetic_prob
                if use_synth:
                    for s in scores:
                        sk = f"synthetic_{s}"
                        if sk in meta:
                            meta[s] = meta[sk]
                for s in scores:
                    if s in meta and isinstance(meta[s], (int, float)):
                        out[s].append(float(meta[s]))
    except Exception as e:
        print(f"[warn] failed reading {tar_path}: {e}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Directory containing tars (also where the cache will be written)")
    p.add_argument("--scores", nargs="+", default=DEFAULT_SCORES)
    p.add_argument("--synthetic-prob", type=float, default=0.5)
    p.add_argument("--max-samples", type=int, default=100000)
    p.add_argument("--num-bins", type=int, default=64)
    p.add_argument("--strategy", default="quantile", choices=["uniform", "quantile", "refined_quantile"])
    p.add_argument("--refine-last-n-bins", type=int, default=0)
    p.add_argument("--refine-factor", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--samples-per-tar", type=int, default=None,
                   help="Cap samples drawn per tar (default: all)")
    args = p.parse_args()

    root = Path(args.root)
    tars = sorted(root.glob("*.tar"))
    if not tars:
        print(f"No tar files in {root}", file=sys.stderr); sys.exit(2)

    # Aim for ~max_samples in total: pick samples_per_tar based on how many tars.
    if args.samples_per_tar is None:
        samples_per_tar = max(1, args.max_samples // len(tars) + 1)
    else:
        samples_per_tar = args.samples_per_tar

    print(f"Scanning {len(tars)} tars with up to {samples_per_tar} samples/tar "
          f"-> target ~{samples_per_tar * len(tars)} samples")

    # Use one process per tar (lightweight). Cap concurrency by --num-workers.
    work = [
        (str(tar), args.scores, args.synthetic_prob, samples_per_tar, args.seed + i)
        for i, tar in enumerate(tars)
    ]

    pooled = {s: [] for s in args.scores}
    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        futs = [ex.submit(_sample_scores_from_tar, w) for w in work]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="tars"):
            local = fut.result()
            for s in args.scores:
                pooled[s].extend(local[s])

    # If we have more than max_samples for any score, sub-sample.
    rng = random.Random(args.seed)
    for s in args.scores:
        if len(pooled[s]) > args.max_samples:
            pooled[s] = rng.sample(pooled[s], args.max_samples)
        print(f"  {s}: {len(pooled[s])} values")

    bins: dict[str, list[float]] = {}
    for s in args.scores:
        vals = pooled[s]
        if not vals:
            print(f"[warn] no values for {s}, skipping")
            continue
        edges = _compute_bins(
            torch.tensor(vals, dtype=torch.float32),
            args.num_bins,
            args.strategy,
            refine_last_n_bins=args.refine_last_n_bins,
            refine_factor=args.refine_factor,
        )
        if edges is None:
            print(f"[warn] _compute_bins returned None for {s}, skipping")
            continue
        bins[s] = edges.tolist()
        print(f"  {s}: {len(bins[s])} edges in [{bins[s][0]:.4f} .. {bins[s][-1]:.4f}]")

    # Compose the filename TextWIDSDataset expects.
    sorted_scores = sorted(args.scores)
    scores_str = "-".join(sorted_scores)
    if args.strategy == "quantile":
        ckpt_name = (
            f"coherence_bins_{args.num_bins}_bins_{scores_str}_"
            f"synthetic_prob_{args.synthetic_prob}.json"
        )
    elif args.strategy == "refined_quantile":
        ckpt_name = (
            f"coherence_bins_{args.num_bins}_bins_{scores_str}_strategy_refined_quantile_"
            f"refine_{args.refine_last_n_bins}_factor_{args.refine_factor}_"
            f"synthetic_prob_{args.synthetic_prob}.json"
        )
    elif args.strategy == "uniform":
        ckpt_name = (
            f"coherence_bins_{args.num_bins}_bins_{scores_str}_strategy_uniform_"
            f"synthetic_prob_{args.synthetic_prob}.json"
        )

    out_path = root / ckpt_name
    out_path.write_text(json.dumps(bins))
    print(f"\nWrote bin cache: {out_path}")


if __name__ == "__main__":
    main()
