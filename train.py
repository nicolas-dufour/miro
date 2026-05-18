import os
import shutil
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor

from miro.callbacks import EMACallback, FixNANinGrad, LogGeneratedImages
from miro.models.diffusion import DiffusionModule
from miro.utils.misc import print0

torch.set_float32_matmul_precision("highest")
OmegaConf.register_new_resolver("eval", eval)


def _resolve_checkpoint(cfg, model):
    """Find checkpoint to resume from, or load init weights."""
    ckpt_dir = Path(cfg.checkpoints.dirpath)
    last_ckpt = ckpt_dir / "last.ckpt"
    init_ckpt = ckpt_dir / "init.ckpt"

    if last_ckpt.exists():
        return last_ckpt

    if init_ckpt.exists():
        ckpt = torch.load(init_ckpt, map_location="cpu", weights_only=False)
        state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        model.load_state_dict(state, strict=False)
        print0("Loaded model from init checkpoint")

    return None


@hydra.main(config_path="configs", config_name="config", version_base=None)
def train(cfg):
    seed_everything(3407, workers=True)

    ckpt_dir = Path(cfg.checkpoints.dirpath)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print0(f"Working directory: {os.getcwd()}")
    shutil.copyfile(Path(".hydra/config.yaml"), ckpt_dir / "config.yaml")

    # Data
    datamodule = hydra.utils.instantiate(cfg.data.datamodule)
    datamodule.setup()

    # Callbacks
    callbacks = [
        hydra.utils.instantiate(cfg.checkpoints),
        hydra.utils.instantiate(cfg.progress_bar),
        EMACallback(
            "network",
            "ema_network",
            decay=cfg.model.ema_decay,
            start_ema_step=cfg.model.start_ema_step,
            init_ema_random=False,
            default_ema_index=cfg.model.get("default_ema_index", 0),
        ),
        LearningRateMonitor(),
        FixNANinGrad(monitor=["train/loss"], set_nan_to_zero=True),
        LogGeneratedImages(
            root_dir=cfg.root_dir,
            num_samples_per_cond=4,
            shape=(cfg.data.in_channels, cfg.data.data_resolution, cfg.data.data_resolution),
            log_every_n_steps=cfg.trainer.val_check_interval,
            cfg_rate=cfg.model.cfg_rate,
            text_embedding_name=cfg.model.text_embedding_name,
            coherence_keys=cfg.model.get("coherence_keys", None),
            coherence_values=cfg.model.get("coherence_values", None),
            uncoherence_values=cfg.model.get("uncoherence_values", None),
            negative_prompts=cfg.model.get("negative_prompts", None),
            batch_size=64,
            debug_mode=False,
        ),
    ]

    # Logger
    dict_config = OmegaConf.to_container(cfg, resolve=True)
    logger = hydra.utils.instantiate(cfg.logger)
    logger._wandb_init.update({
        "config": {"model": dict_config["model"], "data": dict_config["data"]},
    })

    # Model & trainer
    model = DiffusionModule(cfg.model)
    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger, callbacks=callbacks)

    ckpt_path = _resolve_checkpoint(cfg, model)
    trainer.fit(model, datamodule, ckpt_path=ckpt_path, weights_only=False)


if __name__ == "__main__":
    train()
