"""Compute summary scores from GenEval evaluation results."""

import argparse
import os

import numpy as np
import pandas as pd


def summarize(filename):
    with open(os.path.join(os.path.dirname(__file__), "object_names.txt")) as cls_file:
        classnames = [line.strip() for line in cls_file]

    df = pd.read_json(filename, orient="records", lines=True)

    print("Summary")
    print("=======")
    print(f"Total images: {len(df)}")
    print(f"Total prompts: {len(df.groupby('metadata'))}")
    print(f"% correct images: {df['correct'].mean():.2%}")
    print(f"% correct prompts: {df.groupby('metadata')['correct'].any().mean():.2%}")
    print()

    task_scores = []
    print("Task breakdown")
    print("==============")
    for tag, task_df in df.groupby("tag", sort=False):
        task_scores.append(task_df["correct"].mean())
        print(
            f"{tag:<16} = {task_df['correct'].mean():.2%} "
            f"({task_df['correct'].sum()} / {len(task_df)})"
        )
    print()

    overall = np.mean(task_scores)
    print(f"Overall score (avg. over tasks): {overall:.5f}")
    return overall


def main():
    parser = argparse.ArgumentParser(description="GenEval: Summarize evaluation results")
    parser.add_argument("filename", type=str, help="Path to results.jsonl")
    args = parser.parse_args()
    summarize(args.filename)


if __name__ == "__main__":
    main()
