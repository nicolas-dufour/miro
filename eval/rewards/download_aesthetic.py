"""Download Schuhmann's sac+logos+ava1-l14-linearMSE checkpoint into ./models/.

This is the linear MLP head on top of CLIP ViT-L/14 image embeddings that
predicts an aesthetic score. The full predictor combines this MLP with the
OpenAI CLIP ViT-L/14 image encoder (which OpenAI's `clip` package downloads
on first use).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import urlopen

CHECKPOINT_URL = (
    "https://github.com/christophschuhmann/"
    "improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "models",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    target = args.out_dir / "sac+logos+ava1-l14-linearMSE.pth"
    if target.exists():
        print(f"Aesthetic checkpoint already at {target}")
        return

    print(f"Downloading {CHECKPOINT_URL} -> {target}")
    with urlopen(CHECKPOINT_URL) as src, open(target, "wb") as dst:
        while True:
            chunk = src.read(1 << 20)
            if not chunk:
                break
            dst.write(chunk)
    print(f"Wrote {target} ({target.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
