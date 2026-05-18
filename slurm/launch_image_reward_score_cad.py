import argparse
import os
from pathlib import Path

from launch import SlurmExperiment


def parse_mode():
    parser = argparse.ArgumentParser(
        description="Launch Image Reward Score CAD training on 4 nodes of 8 GPUs."
    )
    parser.add_argument("--launch", action="store_true")
    return parser.parse_args()


exp_name = "CC12M_LAION_Aesthetics6_256_ImageRewardScoreCAD"

hydra_args = {
    "experiment": "image_reward_score_cad",
    "computer.devices": 8,
    "computer.num_nodes": 4,
    "computer.precision": "bf16-mixed",
    "computer.progress_bar_refresh_rate": 10,
    "data_dir": os.environ["MIRO_DATA_DIR"],
    "data.datamodule.shard_exclusive_workers": True,
    "experiment_name": exp_name,
}

exp = SlurmExperiment(
    exp_name,
    "image_reward_score_cad",
    num_nodes=4,
    num_gpus_per_node=8,
)
exp.cmd_path = "miro/train.py"


if __name__ == "__main__":
    args = parse_mode()
    exp.build_cmd(hydra_args=hydra_args)
    if args.launch:
        exp.launch()
