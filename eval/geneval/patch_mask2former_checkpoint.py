"""Patch the Mask2Former Swin-S COCO checkpoint from mmdet 2.0 layout to the
mmdet 3.x layout used by ``init_detector``.

The v2.0 checkpoint stores transformer decoder weights as:
    panoptic_head.transformer_decoder.layers.{i}.attentions.0.attn.*   (cross-attn)
    panoptic_head.transformer_decoder.layers.{i}.attentions.1.attn.*   (self-attn)
    panoptic_head.transformer_decoder.layers.{i}.ffns.0.layers.*

mmdet 3.x's ``Mask2FormerTransformerDecoderLayer`` exposes:
    panoptic_head.transformer_decoder.layers.{i}.cross_attn.attn.*
    panoptic_head.transformer_decoder.layers.{i}.self_attn.attn.*
    panoptic_head.transformer_decoder.layers.{i}.ffn.layers.*

Mask2Former's operation order is cross-attention BEFORE self-attention, so
``attentions[0]`` (first attention in the old per-layer ModuleList) is the
cross-attention. The pixel-decoder encoder layers only have one attention
(``attentions.0`` -> ``self_attn``).

The patch is idempotent and writes the result in-place (backing up the
original to ``<name>_original.pth`` if no backup exists yet).
"""
from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path

import torch


def rename_key(k: str) -> str:
    new_k = k
    if ".transformer_decoder.layers." in k:
        new_k = new_k.replace(".attentions.0.attn.", ".cross_attn.attn.")
        new_k = new_k.replace(".attentions.1.attn.", ".self_attn.attn.")
        new_k = new_k.replace(".ffns.0.layers.", ".ffn.layers.")
    if ".pixel_decoder.encoder.layers." in k:
        new_k = new_k.replace(".attentions.0.", ".self_attn.")
        new_k = new_k.replace(".ffns.0.", ".ffn.")
    return new_k


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()

    ckpt_path: Path = args.checkpoint
    if not ckpt_path.exists():
        sys.exit(f"checkpoint not found: {ckpt_path}")

    backup_path = ckpt_path.with_name(ckpt_path.stem + "_original" + ckpt_path.suffix)
    if not backup_path.exists():
        shutil.copy2(ckpt_path, backup_path)
        print(f"backed up {ckpt_path.name} -> {backup_path.name}")

    sd = torch.load(backup_path, map_location="cpu", weights_only=False)
    state = sd["state_dict"]
    new_state = {}
    renamed = 0
    for k, v in state.items():
        new_k = rename_key(k)
        if new_k != k:
            renamed += 1
        new_state[new_k] = v
    sd["state_dict"] = new_state
    torch.save(sd, ckpt_path)
    print(f"renamed {renamed} of {len(state)} keys, wrote {ckpt_path}")


if __name__ == "__main__":
    main()
