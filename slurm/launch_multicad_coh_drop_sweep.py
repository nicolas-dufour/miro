"""Launch 4 MIRO multi-CAD synth training jobs sweeping coherence_dropout.

Each job is 2 nodes x 8 GPUs (16 GPUs total), fp16-mixed precision, matching
the legacy CAD GT setup. Variants only differ in
``model.network.instance.coherence_dropout``.

Submit with::

    uv run --extra train python miro/slurm/launch_multicad_coh_drop_sweep.py --launch
"""
import os
import argparse

from launch import SlurmExperiment


def parse_mode():
    parser = argparse.ArgumentParser(
        description="Launch coherence_dropout sweep on 2 nodes x 8 GPUs per job."
    )
    parser.add_argument("--launch", action="store_true")
    return parser.parse_args()


COHERENCE_DROPOUTS = [0.25, 0.5, 0.77, 0.9]


def _slug(p: float) -> str:
    # 0.25 -> "025", 0.5 -> "050", 0.77 -> "077", 0.9 -> "090"
    return f"{int(round(p * 100)):03d}"


def main() -> None:
    args = parse_mode()
    for p in COHERENCE_DROPOUTS:
        exp_name = f"MIRO_coh_drop_{_slug(p)}_16gpu"
        hydra_args = {
            "experiment": "multi_cad_synth",
            "computer.devices": 8,
            "computer.num_nodes": 2,
            "computer.precision": "16-mixed",
            "computer.progress_bar_refresh_rate": 10,
            "data_dir": os.environ["MIRO_DATA_DIR"],
            "data.datamodule.shard_exclusive_workers": True,
            "experiment_name": exp_name,
            "model.network.instance.coherence_dropout": p,
        }
        exp = SlurmExperiment(
            exp_name,
            f"multicad_coh_drop_{_slug(p)}",
            num_nodes=2,
            num_gpus_per_node=8,
        )
        exp.cmd_path = "miro/train.py"
        exp.build_cmd(hydra_args=hydra_args)
        if args.launch:
            exp.launch()


if __name__ == "__main__":
    main()
