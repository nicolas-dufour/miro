"""Convert a PartiPrompts-style TSV (Prompt, Category, Challenge, Note) to a
JSONL file shaped for ``miro/scripts/generate_geneval.py``: one line per
prompt with at least a ``prompt`` key.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tsv", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out = args.out or args.tsv.with_suffix(".jsonl")
    df = pd.read_csv(args.tsv, sep="\t")
    if "Prompt" not in df.columns:
        raise SystemExit(f"{args.tsv} has no 'Prompt' column (got {list(df.columns)})")

    with open(out, "w") as f:
        for _, row in df.iterrows():
            entry = {"prompt": row["Prompt"]}
            for key in ("Category", "Challenge", "Note"):
                if key in row and pd.notna(row[key]):
                    entry[key.lower()] = row[key]
            f.write(json.dumps(entry) + "\n")
    print(f"Wrote {len(df)} prompts to {out}")


if __name__ == "__main__":
    main()
