import contextlib
import math
import random

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from PIL import Image
from pytorch_lightning import Callback
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from miro.data.datamodule import dict_collate_and_pad
from miro.utils.misc import print0


def _get_loggers_list(trainer_logger):
    """Get a list of loggers from trainer.logger."""
    if trainer_logger is None:
        return []
    if isinstance(trainer_logger, list):
        return trainer_logger
    if hasattr(trainer_logger, "_loggers"):
        return trainer_logger._loggers
    if hasattr(trainer_logger, "loggers"):
        return trainer_logger.loggers
    return [trainer_logger]


def _is_wandb_logger(logger):
    return logger.__class__.__name__ == "WandbLogger"


def _is_trackio_logger(logger):
    class_name = logger.__class__.__name__
    module_name = logger.__class__.__module__ or ""
    return class_name == "TrackioLogger" or "trackio" in module_name.lower()


@contextlib.contextmanager
def temp_seed(seed):
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)


class TextCondPromptBed(Dataset):
    def __init__(
        self,
        path,
        num_samples_per_cond,
        text_embedding_name=None,
        generator=None,
        shape=(4, 32, 32),
    ):
        self.path = path
        self.num_samples_per_cond = num_samples_per_cond
        self.metadata = pd.read_csv(path / "metadata.csv")
        self.num_samples = len(self.metadata) * num_samples_per_cond
        self.text_embedding_name = text_embedding_name
        self.generator = generator
        self.x_N = torch.randn(
            self.num_samples, *shape, generator=self.generator, dtype=torch.float32
        )

    def __getitem__(self, index):
        metadata = self.metadata.iloc[index // self.num_samples_per_cond]
        text = metadata["text"]
        if self.text_embedding_name is None:
            return {"x_N": self.x_N[index], "text": text}
        embedding = torch.from_numpy(
            np.load(
                self.path
                / f"{self.text_embedding_name}_embeddings"
                / metadata["file_name"]
            )
        )
        return {
            "x_N": self.x_N[index],
            "text": text,
            self.text_embedding_name: embedding,
        }

    def __len__(self):
        return self.num_samples


class LogGeneratedImages(Callback):
    def __init__(
        self,
        root_dir,
        num_samples_per_cond: int = 4,
        shape=(4, 32, 32),
        log_every_n_steps: int = 25000,
        cfg_rate=7.0,
        text_embedding_name="flan_t5_xl",
        coherence_keys=None,
        coherence_values=None,
        uncoherence_values=None,
        negative_prompts=None,
        batch_size=64,
        debug_mode=False,
        log_steps=None,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.log_steps = log_steps
        self.num_samples_per_cond = num_samples_per_cond
        assert math.sqrt(num_samples_per_cond) % 1 == 0
        self.sqrt_num_samples = int(math.sqrt(num_samples_per_cond))
        self.shape = shape
        self.log_every_n_steps = log_every_n_steps
        self.batch_size = min(batch_size, 64)
        self.cfg_rate = cfg_rate
        self.text_embedding_name = text_embedding_name
        self.coherence_keys = coherence_keys
        self.coherence_values = coherence_values
        self.uncoherence_values = uncoherence_values
        self.debug_mode = debug_mode
        self.last_logged_step = -1

        if negative_prompts == "random_prompt":
            from miro.utils.assets import load_random_prompt_bank
            self.negative_prompts = torch.from_numpy(
                load_random_prompt_bank(text_embedding_name)
            )
        else:
            self.negative_prompts = None

    def on_train_start(self, trainer, pl_module):
        try:
            self.world_size = torch.distributed.get_world_size()
        except Exception:
            self.world_size = 1

        generator = torch.Generator(device="cpu").manual_seed(3407)
        testbed_path = f"{self.root_dir}/miro/datasets/text_prompt_testbed"
        self.prompt_bed = TextCondPromptBed(
            path=__import__("pathlib").Path(testbed_path),
            num_samples_per_cond=self.num_samples_per_cond,
            text_embedding_name=self.text_embedding_name,
            generator=generator,
            shape=self.shape,
        )

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if (
            pl_module.global_step % self.log_every_n_steps == 1
            and not trainer.sanity_checking
            and pl_module.global_step > self.last_logged_step
        ):
            self.last_logged_step = pl_module.global_step
            print0("Logging images")
            pl_module.eval()
            self._log_images(trainer, pl_module, prefix="val")
            pl_module.train()

    def _log_images(self, trainer, pl_module, prefix="val"):
        if trainer.sanity_checking:
            return

        loggers = _get_loggers_list(
            trainer.loggers if hasattr(trainer, "loggers") else trainer.logger
        )
        if (
            isinstance(self.negative_prompts, torch.Tensor)
            and self.negative_prompts.device != pl_module.device
        ):
            self.negative_prompts = self.negative_prompts.to(pl_module.device)

        generator = torch.Generator(pl_module.device).manual_seed(3407)

        steps_list = self.log_steps if self.log_steps is not None else [None]

        collate_fn = dict_collate_and_pad(
            [self.text_embedding_name], max_length=256
        )
        if self.world_size > 1:
            dataloader = DataLoader(
                self.prompt_bed,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4,
                sampler=DistributedSampler(self.prompt_bed, shuffle=False, drop_last=False),
                collate_fn=collate_fn,
            )
        else:
            dataloader = DataLoader(
                self.prompt_bed,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4,
                collate_fn=collate_fn,
            )

        for num_steps in steps_list:
            step_suffix = f" {num_steps} steps" if num_steps is not None else ""
            results = {}
            batch_count = 0

            for batch in tqdm(dataloader, desc="Generating images"):
                if self.debug_mode and batch_count >= 2:
                    print0("Debug mode: limiting image logging to first 2 batches")
                    break
                batch_count += 1

                with torch.no_grad():
                    x_k = batch["x_N"]
                    name = batch["text"]
                    del batch["text"]
                    del batch["x_N"]
                    cond = {k: v.to(pl_module.device) for k, v in batch.items()}
                    x_k = x_k.to(pl_module.device)

                    sample_kwargs = {
                        "x_N": x_k,
                        "cond": cond,
                        "stage": prefix,
                        "generator": generator,
                        "cfg": self.cfg_rate,
                        "unconfident_prompt": self.negative_prompts,
                        "coherence_keys": self.coherence_keys,
                        "coherence_values": self.coherence_values,
                        "uncoherence_values": self.uncoherence_values,
                    }
                    if num_steps is not None:
                        sample_kwargs["num_steps"] = num_steps

                    sampled_images = pl_module.sample(**sample_kwargs)

                    if self.world_size > 1:
                        sampled_images = pl_module.all_gather(sampled_images).flatten(0, 1)
                        name_list = [None for _ in range(self.world_size)]
                        torch.distributed.all_gather_object(name_list, name)
                        name = [item for sublist in name_list for item in sublist]

                for n, img in zip(name, sampled_images):
                    if n not in results:
                        results[n] = []
                    if len(results[n]) < self.num_samples_per_cond:
                        results[n].append(img)

            if pl_module.global_rank == 0:
                pil_images = {}
                for name, images in results.items():
                    if isinstance(images, list):
                        if isinstance(images[0], np.ndarray):
                            images = np.stack(images, axis=0)
                        else:
                            images = torch.stack(images, dim=0)
                    images = rearrange(
                        images,
                        "(b1 b2) h w c -> (b1 h) (b2 w) c",
                        b1=self.sqrt_num_samples,
                        b2=self.sqrt_num_samples,
                    )
                    if isinstance(images, torch.Tensor):
                        images = images.cpu().numpy()
                    if isinstance(images, np.ndarray):
                        if images.dtype != np.uint8:
                            images = np.clip(images * 255, 0, 255).astype(np.uint8)
                        images = Image.fromarray(images)

                    truncated_name = name[:50] + "..." if len(name) > 50 else name
                    key = f"{prefix}/Images{step_suffix}/Samples text {truncated_name}"
                    pil_images[key] = images

                for logger in loggers:
                    if _is_wandb_logger(logger):
                        import wandb

                        logs = {k: wandb.Image(v, file_type="jpg") for k, v in pil_images.items()}
                        logger.experiment.log({
                            **logs,
                            "trainer/global_step": pl_module.global_step,
                        })
                    elif _is_trackio_logger(logger):
                        import trackio

                        _ = logger.experiment
                        step = (
                            int(pl_module.global_step)
                            if hasattr(pl_module.global_step, "item")
                            else pl_module.global_step
                        )
                        logs = {k: trackio.Image(value=v) for k, v in pil_images.items()}
                        trackio.log({**logs, "trainer/global_step": step})
