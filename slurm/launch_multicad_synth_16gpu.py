"""Launch MIRO multi-CAD synth training on 2 nodes x 8 GPUs (16 total).

Matches the legacy CAD architecture used to train
``miro/checkpoints_ground_truth/CC12M_256_RIN_small_flow_multi_cad_synth_synth_p50``
via the ``multi_cad_synth`` experiment with
``use_cond_rin_block=true``, ``concat_cond_token_to_latents=false``, and
``use_self_conditioning=true`` baked into ``miro/configs/experiment/multi_cad.yaml``.

Submit with::

    cd miro/slurm && python launch_multicad_synth_16gpu.py --launch
"""
import os
import argparse

from launch import SlurmExperiment


def parse_mode():
    parser = argparse.ArgumentParser(
        description="Launch Multi-CAD Synth training on 2 nodes of 8 GPUs (16 GPUs total)."
    )
    parser.add_argument("--launch", action="store_true")
    return parser.parse_args()


exp_name = "MIRO_synth_legacy_16gpu_v3"

hydra_args = {
    "experiment": "multi_cad_synth",
    "computer.devices": 8,
    "computer.num_nodes": 2,
    "computer.precision": "16-mixed",
    "computer.progress_bar_refresh_rate": 10,
    "data_dir": os.environ["MIRO_DATA_DIR"],
    "data.datamodule.shard_exclusive_workers": True,
    "experiment_name": exp_name,
}

exp = SlurmExperiment(
    exp_name,
    "multicad_synth_16gpu",
    num_nodes=2,
    num_gpus_per_node=8,
)
exp.cmd_path = "miro/train.py"


if __name__ == "__main__":
    args = parse_mode()
    exp.build_cmd(hydra_args=hydra_args)
    if args.launch:
        exp.launch()
