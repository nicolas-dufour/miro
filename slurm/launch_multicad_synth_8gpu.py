"""Launch the main MIRO recipe on 1 node x 8 H100 GPUs with gradient accumulation.

Single-node fallback for environments that don't have two nodes available.
Keeps the same effective batch size (1024 samples / global step) as the
16-GPU reference run by setting ``trainer.accumulate_grad_batches=2``, i.e.
two forward/backward micro-batches before each optimizer step.

Expected wall-clock impact vs. the 16-GPU run: roughly **2x longer** per step
since we're doing twice the work on half the hardware (modulo small fixed
overheads). Always confirm with a short measurement run before committing
to a full training.

Submit with::

    cd miro/slurm && python launch_multicad_synth_8gpu.py --launch

Time-limited test (15 min) for throughput measurement::

    cd miro/slurm && python launch_multicad_synth_8gpu.py --launch --test
"""
import os
import argparse

from launch import SlurmExperiment


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch MIRO multi-CAD synth training on 1 node of 8 GPUs (8 GPUs total)."
    )
    parser.add_argument("--launch", action="store_true")
    parser.add_argument(
        "--test", action="store_true",
        help="Submit a 15-minute timed run, intended for throughput measurement.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    exp_name = "MIRO_synth_8gpu_test" if args.test else "MIRO_synth_8gpu"

    hydra_args = {
        "experiment": "multi_cad_synth",
        "computer.devices": 8,
        "computer.num_nodes": 1,
        "computer.precision": "16-mixed",
        "computer.progress_bar_refresh_rate": 10,
        "data_dir": os.environ["MIRO_DATA_DIR"],
        "data.datamodule.shard_exclusive_workers": True,
        # Keep effective global batch size at 1024 (vs. 16 GPUs * 64 = 1024).
        # ``+`` prefix appends the key — Hydra runs in struct mode here and the
        # base trainer config does not declare ``accumulate_grad_batches``.
        "+trainer.accumulate_grad_batches": 2,
        # The base config sets ``static_graph: true``; with the 10 %
        # self-conditioning skip in ``FlowMatchingLoss`` this trips a DDP
        # assertion under gradient accumulation. Relax those two flags.
        "trainer.strategy.static_graph": "false",
        "trainer.strategy.find_unused_parameters": "true",
        "experiment_name": exp_name,
    }

    exp = SlurmExperiment(
        exp_name,
        "multicad_synth_8gpu",
        num_nodes=1,
        num_gpus_per_node=8,
        time="00:15:00" if args.test else None,
    )
    exp.cmd_path = "miro/train.py"
    exp.build_cmd(hydra_args=hydra_args)
    if args.launch:
        exp.launch()


if __name__ == "__main__":
    main()
