"""Precompute FLAN-T5-XL embeddings for every prompt in metadata.csv.

Reads `metadata.csv` (columns: file_name, text), encodes each prompt with the
FLAN-T5-XL text encoder, and writes one `.npy` per prompt into
`flan_t5_xl_embeddings/` (shape: (n_tokens, 2048), float32, trimmed to the
real token length by attention mask).

These embeddings are consumed by `TextCondPromptBed` in
`miro/callbacks/log_images.py` to produce the qualitative prompt grid logged
during training.

Usage (from the repo root):
    uv run --extra train python miro/datasets/text_prompt_testbed/precompute_logging_embeddings.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, T5EncoderModel

MODEL_NAME = "google/flan-t5-xl"
MAX_LENGTH = 2048


def main():
    testbed = Path(__file__).resolve().parent
    out_dir = testbed / "flan_t5_xl_embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(testbed / "metadata.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = T5EncoderModel.from_pretrained(MODEL_NAME).to(device).eval()

    with torch.no_grad():
        for _, row in metadata.iterrows():
            tok = tokenizer(
                row["text"],
                return_tensors="pt",
                padding="longest",
                max_length=MAX_LENGTH,
                truncation=True,
            )
            tok = {k: v.to(device) for k, v in tok.items()}
            length = int(tok["attention_mask"].sum().item())
            emb = model(**tok).last_hidden_state[0, :length].float().cpu().numpy()
            np.save(out_dir / row["file_name"], emb)
            print(f"{row['file_name']}: {emb.shape}")


if __name__ == "__main__":
    main()
