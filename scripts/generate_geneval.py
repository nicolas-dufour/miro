"""
Generate images for GenEval evaluation using a trained Miro diffusion model.

Usage:
    python scripts/generate_geneval.py \
        --checkpoint checkpoints/CC12M_.../epoch=14-step=250000.ckpt \
        --config checkpoints/CC12M_.../config.yaml \
        --outdir geneval_images/step250k \
        --n-samples 4 --steps 50 --cfg 7.0
"""

import argparse
import json
import os
import random
import string
import sys

import numpy as np
import PIL.Image
import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer

# Add miro root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MIRO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(MIRO_ROOT))

from miro.models.diffusion import DiffusionModule


def encode_prompts(prompts, tokenizer, encoder, device, max_length=256):
    """Encode text prompts using flan-t5-xl, matching precomputed embedding format.

    Precomputed embeddings are variable-length (trimmed to actual tokens), stored
    as float16, then zero-padded during collation. We replicate that exactly:
      1. Tokenize with padding="longest" (no artificial padding)
      2. Encode → last_hidden_state in bfloat16
      3. Zero-pad to max_length with a boolean mask
    """
    tokens = tokenizer(
        prompts,
        padding="longest",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids.to(device)
    attention_mask = tokens.attention_mask.to(device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        raw_embeddings = encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state.detach().half()

    # Zero-pad to max_length (same as dict_collate_and_pad in training)
    batch_size = raw_embeddings.shape[0]
    seq_len = raw_embeddings.shape[1]
    embed_dim = raw_embeddings.shape[2]
    padded_embeddings = torch.zeros(
        batch_size, max_length, embed_dim,
        dtype=raw_embeddings.dtype, device=device,
    )
    padded_mask = torch.zeros(batch_size, max_length, dtype=torch.bool, device=device)
    actual_len = min(seq_len, max_length)
    padded_embeddings[:, :actual_len] = raw_embeddings[:, :actual_len]
    padded_mask[:, :actual_len] = attention_mask[:, :actual_len].bool()

    return padded_embeddings, padded_mask


def generate_random_string():
    """Generate a random string for negative prompting."""
    words = []
    for _ in range(random.randint(3, 6)):
        word = "".join(
            random.choice(string.ascii_lowercase) for _ in range(random.randint(3, 7))
        )
        words.append(word)
    return " ".join(words)


def load_model(config_path, checkpoint_path, device):
    """Load miro model from config + checkpoint using Lightning's load_from_checkpoint."""
    cfg = OmegaConf.load(config_path)

    # The saved config uses ``${root_dir}`` (= ``${hydra:runtime.cwd}``) for a
    # few asset paths — set it to the miro repo root so anything that still
    # references it resolves consistently outside Hydra.
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    cfg.root_dir = MIRO_ROOT
    OmegaConf.resolve(cfg)

    # Disable compilation for inference (faster loading)
    cfg.model.compile = False

    model = DiffusionModule.load_from_checkpoint(
        checkpoint_path, cfg=cfg.model, strict=False, weights_only=False,
    )
    model.eval()
    model.to(device)
    return model, cfg


def main():
    parser = argparse.ArgumentParser(description="Generate images for GenEval")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None,
                        help="Config YAML (default: config.yaml next to checkpoint)")
    parser.add_argument("--metadata-file", type=str, default=None,
                        help="GenEval metadata JSONL (default: eval/geneval/prompts/evaluation_metadata.jsonl)")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Output directory (default: <checkpoint_dir>/geneval_images)")
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=7.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Resolve defaults
    ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    if args.config is None:
        args.config = os.path.join(ckpt_dir, "config.yaml")
    if args.metadata_file is None:
        args.metadata_file = os.path.join(
            MIRO_ROOT, "eval/geneval/prompts/evaluation_metadata.jsonl"
        )
    if args.outdir is None:
        ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        args.outdir = os.path.join(ckpt_dir, "geneval_images", ckpt_name)

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device)
    seed_everything(args.seed)

    # Load prompts
    with open(args.metadata_file) as f:
        prompts_data = [json.loads(line) for line in f]
    print(f"Loaded {len(prompts_data)} prompts from {args.metadata_file}")

    # Load text encoder
    print("Loading flan-t5-xl text encoder...")
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
    encoder = T5EncoderModel.from_pretrained("google/flan-t5-xl").to(device).eval()

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, cfg = load_model(args.config, args.checkpoint, device)
    text_emb_name = cfg.model.text_embedding_name  # "flan_t5_xl"
    shape = (
        cfg.data.in_channels,
        cfg.data.data_resolution,
        cfg.data.data_resolution,
    )

    # Read coherence/guidance params from config
    coherence_keys = OmegaConf.to_container(cfg.model.get("coherence_keys", None))
    coherence_values = OmegaConf.to_container(
        cfg.model.get("coherence_values", {"clip_score_coherence": 1.0})
    )
    uncoherence_values = OmegaConf.to_container(
        cfg.model.get("uncoherence_values", {"clip_score_uncoherence": 0.0})
    )
    guidance_type = cfg.model.get("guidance_type", "constant")
    negative_prompts_mode = cfg.model.get("negative_prompts", None)

    # Prepare negative prompt embeddings (loaded from the bundled package asset)
    unconfident_prompt = None
    if negative_prompts_mode == "random_prompt":
        from miro.utils.assets import asset_path

        random_neg_path = asset_path(f"{text_emb_name}_random.npy")
        if random_neg_path.exists():
            unconfident_prompt = torch.from_numpy(np.load(random_neg_path)).to(device)
            print(f"Loaded random negative embeddings from {random_neg_path}")
        else:
            print(f"Warning: random negative embeddings not found at {random_neg_path}")

    print(f"Coherence keys: {coherence_keys}")
    print(f"Coherence values: {coherence_values}")
    print(f"Guidance type: {guidance_type}")
    print(f"Negative prompts: {negative_prompts_mode}")

    # Save generation hparams
    hparams = {
        "checkpoint": args.checkpoint,
        "n_samples": args.n_samples,
        "steps": args.steps,
        "cfg": args.cfg,
        "seed": args.seed,
        "coherence_keys": coherence_keys,
        "coherence_values": coherence_values,
        "uncoherence_values": uncoherence_values,
        "guidance_type": guidance_type,
        "negative_prompts": negative_prompts_mode,
    }
    with open(os.path.join(args.outdir, "hparams.json"), "w") as f:
        json.dump(hparams, f, indent=2)

    # Prepare per-prompt output structure
    num_prompts = len(prompts_data)
    sample_counts = [0] * num_prompts
    prompt_dirs = []
    samples_dirs = []

    for index, metadata in enumerate(prompts_data):
        prompt_dir = os.path.join(args.outdir, f"{index:05d}")
        samples_dir = os.path.join(prompt_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)
        with open(os.path.join(prompt_dir, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)
        prompt_dirs.append(prompt_dir)
        samples_dirs.append(samples_dir)

    # Batch across prompts (same batching strategy as reference)
    remaining_total = num_prompts * args.n_samples
    pbar = tqdm(total=remaining_total, desc="Generating images")

    while remaining_total > 0:
        batch_indices = []
        batch_prompts = []
        for idx in range(num_prompts):
            if sample_counts[idx] < args.n_samples:
                batch_indices.append(idx)
                batch_prompts.append(prompts_data[idx]["prompt"])
                if len(batch_indices) == args.batch_size:
                    break
        if not batch_indices:
            break

        bs = len(batch_indices)

        # Encode prompts
        embeddings, mask = encode_prompts(
            batch_prompts, tokenizer, encoder, device
        )

        with torch.no_grad():
            images = model.sample(
                batch_size=bs,
                shape=shape,
                cond={
                    f"{text_emb_name}_embeddings": embeddings,
                    f"{text_emb_name}_mask": mask,
                },
                cfg=args.cfg,
                num_steps=args.steps,
                stage="test",
                guidance_type=guidance_type,
                coherence_keys=coherence_keys,
                coherence_values=coherence_values,
                uncoherence_values=uncoherence_values,
                unconfident_prompt=unconfident_prompt,
            )

        # Save images — model.sample() returns (B, H, W, 3) uint8 numpy via postprocessing
        for b in range(len(batch_indices)):
            pidx = batch_indices[b]
            save_idx = sample_counts[pidx]
            img_data = images[b]
            # Handle both tensor (B, C, H, W) and numpy (B, H, W, 3) outputs
            if isinstance(img_data, torch.Tensor):
                img_data = img_data.permute(1, 2, 0).cpu().numpy()
            if img_data.dtype != np.uint8:
                img_data = np.clip(img_data, 0, 255).astype(np.uint8)
            img = PIL.Image.fromarray(img_data)
            img.save(os.path.join(samples_dirs[pidx], f"{save_idx:04d}.png"))
            sample_counts[pidx] += 1
            remaining_total -= 1

        pbar.update(len(batch_indices))

    pbar.close()
    print(f"Done. Images saved to: {args.outdir}")

    # Free text encoder VRAM
    del encoder, tokenizer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
