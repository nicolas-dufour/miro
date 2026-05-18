# Text prompt testbed

Fixed set of 80 prompts used by the training-time image logging callback
([miro/callbacks/log_images.py](../../callbacks/log_images.py)) to produce a
consistent qualitative grid across runs.

## Files

- `prompts.txt` — one prompt per line.
- `metadata.csv` — `(file_name, text)` pairs that pair each prompt with its
  embedding filename.
- `flan_t5_xl_embeddings/` — per-prompt FLAN-T5-XL embeddings as `.npy`
  (shape `(n_tokens, 2048)`, float32). **Not committed**; precompute locally.

## Precomputing the embeddings

From the repo root:

```bash
uv run --extra train python miro/datasets/text_prompt_testbed/precompute_logging_embeddings.py
```

This downloads `google/flan-t5-xl` via `transformers`, encodes each row of
`metadata.csv`, and writes one `.npy` per prompt into `flan_t5_xl_embeddings/`.
Requires a CUDA GPU with enough memory to hold the T5-XL encoder (~10 GB).
