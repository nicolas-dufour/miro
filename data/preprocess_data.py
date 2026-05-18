"""Multi-stage preprocessing pipeline for raw webdataset tars (cc12m-style).

Stages (designed to run separately so models don't compete for GPU memory):
  1. rewards  – all reward scores except VQAScore for both original and
                synthetic captions.  Writes a CSV per tar.
  2. vqa      – VQAScore for original + synthetic captions.  Updates the CSV.
  3. vae      – SDXL VAE encoding (256 & 512) + T5 text embeddings for
                original and synthetic captions.  Reads the CSV, reads the
                source tar, and writes a new output tar with everything.

Each stage can be run on a single tar (--shard_id / --shard_range) for easy
SLURM parallelisation, or it can iterate over all tars sequentially.

Raw data format (cc12m-style):
  {shard_id}.tar  containing  {key}.jpg  {key}.json  {key}.txt

Synthetic captions are provided via --synthetic_captions (TSV file) or read
from the sample JSON if already present (keys: synthetic_caption,
short_synthetic_caption).
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import csv
import io
import json
import logging
import os
from functools import partial

import numpy as np
import torch
import webdataset as wds
from diffusers import AutoencoderKL
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    CLIPModel,
    CLIPProcessor,
    T5EncoderModel,
)

from miro.utils.image_processing import CenterCrop

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global model handles (loaded lazily)
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

text_tokenizer = None
text_model = None
loaded_text_model_name = None
vae_model = None
aesthetic_scorer = None
image_reward_model = None
pickscore_processor = None
pickscore_model = None
vqa_processor = None
clip_model = None
clip_tokenizer = None
clip_image_transform = None
sciscore_processor = None
sciscore_model = None
hpsv2 = None

# ---------------------------------------------------------------------------
# Image transforms
# ---------------------------------------------------------------------------
vae_image_transforms_256 = transforms.Compose([
    CenterCrop(ratio="1:1"),
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5),
])

vae_image_transforms_512 = transforms.Compose([
    CenterCrop(ratio="1:1"),
    transforms.Resize(512),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5),
])

# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_text_model(model_name: str = "google/flan-t5-xl"):
    global text_tokenizer, text_model, loaded_text_model_name
    if text_tokenizer is None or loaded_text_model_name != model_name:
        text_tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model_name == "google/flan-t5-xl":
            text_model = T5EncoderModel.from_pretrained(model_name)
        else:
            text_model = AutoModel.from_pretrained(model_name)
        text_model = text_model.to(device).eval()
        loaded_text_model_name = model_name
        logger.info("Text model loaded: %s", model_name)


def load_vae_model():
    global vae_model
    if vae_model is None:
        vae_model = AutoencoderKL.from_pretrained(
            "stabilityai/sdxl-vae", device="cuda:0"
        ).to(device)
        vae_model.requires_grad_(False)
        vae_model.eval()
        logger.info("VAE model loaded")


def load_reward_models(
    load_clip=False,
    load_aesthetic_score=False,
    load_image_reward=False,
    load_pickscore=False,
    load_hpsv2=False,
    load_sciscore=False,
):
    global clip_model, clip_tokenizer, clip_image_transform
    global aesthetic_scorer, image_reward_model
    global pickscore_processor, pickscore_model
    global hpsv2, sciscore_processor, sciscore_model

    if load_clip and clip_model is None:
        clip_model = AutoModel.from_pretrained(
            "jinaai/jina-clip-v2", torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to(device).eval()
        clip_tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-clip-v2")
        clip_image_transform = AutoImageProcessor.from_pretrained(
            "jinaai/jina-clip-v2", trust_remote_code=True
        )
        logger.info("CLIP model loaded")

    if load_hpsv2:
        import miro.utils.rewards.hpsv2 as hpsv2_module
        hpsv2 = hpsv2_module
        logger.info("HPSv2 loaded")

    if load_aesthetic_score and aesthetic_scorer is None:
        from miro.utils.rewards.aesthetic import AestheticScorer
        aesthetic_scorer = AestheticScorer().to(device).eval()
        logger.info("Aesthetic scorer loaded")

    if load_image_reward and image_reward_model is None:
        from miro.utils.rewards.image_reward import load as load_ir
        image_reward_model = load_ir("ImageReward-v1.0", device=device)
        image_reward_model.eval()
        logger.info("Image reward model loaded")

    if load_pickscore and pickscore_processor is None:
        pickscore_processor = CLIPProcessor.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        )
        pickscore_model = CLIPModel.from_pretrained("yuvalkirstain/PickScore_v1").to(device).eval()
        logger.info("PickScore model loaded")

    if load_sciscore and sciscore_processor is None:
        sciscore_processor = AutoProcessor.from_pretrained("Jialuo21/SciScore")
        sciscore_model = AutoModel.from_pretrained("Jialuo21/SciScore").eval().to(device)
        logger.info("SciScore model loaded")


def load_vqa_model():
    global vqa_processor
    if vqa_processor is None:
        from miro.utils.rewards.vqa_scores import VQAScores
        vqa_processor = VQAScores()
        logger.info("VQA processor loaded")


# ---------------------------------------------------------------------------
# Collation / helpers
# ---------------------------------------------------------------------------
VARIABLE_LENGTH_KEYS = {
    "vae_embeddings_mean_256", "vae_embeddings_std_256",
    "vae_embeddings_mean_512", "vae_embeddings_std_512",
}


def dict_collate(batch):
    out = {}
    if not batch:
        return out
    if isinstance(batch[0], dict):
        for key in batch[0]:
            vals = [d[key] for d in batch if key in d]
            if not vals:
                continue
            if key in VARIABLE_LENGTH_KEYS or key == "json":
                out[key] = vals
            elif isinstance(vals[0], Image.Image):
                out[key] = vals
            else:
                try:
                    out[key] = torch.utils.data.dataloader.default_collate(vals)
                except Exception:
                    out[key] = vals
        return out
    elif isinstance(batch[0], Image.Image):
        return list(batch)
    else:
        try:
            return torch.utils.data.dataloader.default_collate(batch)
        except Exception:
            return batch


def log_and_continue(exn):
    return True


def _clean(text):
    return text.replace("<PERSON>", "")


# ---------------------------------------------------------------------------
# Sample decoder for webdataset
# ---------------------------------------------------------------------------
def decode_sample(sample):
    key = sample["__key__"]
    try:
        pil = Image.open(io.BytesIO(sample["jpg"])).convert("RGB")
        txt = sample["txt"].decode("utf-8") if isinstance(sample["txt"], bytes) else sample["txt"]
        meta = json.loads(sample["json"].decode("utf-8") if isinstance(sample["json"], bytes) else sample["json"])
        return {
            "__key__": key,
            "image": pil,
            "txt": txt,
            "json": meta,
            "orig_image_bytes": sample["jpg"],
        }
    except Exception as e:
        logger.warning("Error decoding sample %s: %s", key, e)
        return None


# ---------------------------------------------------------------------------
# CSV I/O helpers
# ---------------------------------------------------------------------------
def csv_path_for_tar(csv_dir, tar_path):
    """Return the CSV path corresponding to a tar file."""
    return csv_dir / (Path(tar_path).stem + "_scores.csv")


def read_csv_scores(csv_file):
    """Read per-sample scores from a CSV into a dict keyed by sample key."""
    scores = {}
    if not csv_file.exists():
        return scores
    with open(csv_file, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row.pop("key")
            # Convert numeric strings back to floats
            scores[key] = {k: float(v) if v != "" else None for k, v in row.items()}
    return scores


def write_csv_scores(csv_file, scores_dict):
    """Write per-sample scores dict to CSV. Keys become columns."""
    if not scores_dict:
        return
    all_keys = list(next(iter(scores_dict.values())).keys())
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["key"] + sorted(all_keys))
        writer.writeheader()
        for sample_key in sorted(scores_dict.keys()):
            row = {"key": sample_key}
            row.update(scores_dict[sample_key])
            writer.writerow(row)


def update_csv_scores(csv_file, new_scores):
    """Merge new_scores into an existing CSV (or create it)."""
    existing = read_csv_scores(csv_file)
    for key, vals in new_scores.items():
        if key in existing:
            existing[key].update(vals)
        else:
            existing[key] = vals
    write_csv_scores(csv_file, existing)


# ---------------------------------------------------------------------------
# Stage 1: Rewards (except VQA)
# ---------------------------------------------------------------------------
def stage_rewards(
    tar_path,
    csv_dir,
    batch_size=64,
    caption_df=None,
    compute_clip_score=True,
    compute_aesthetic_score=True,
    compute_image_reward=True,
    compute_pickscore=True,
    compute_hpsv2=True,
    compute_sciscore=True,
    compute_synthetic=True,
):
    """Compute all reward scores (except VQA) for original + synthetic captions.

    Saves scores to a CSV per tar.
    """
    csv_file = csv_path_for_tar(csv_dir, tar_path)
    tar_stem = Path(tar_path).stem

    # Build webdataset pipeline
    dataset = wds.DataPipeline(
        wds.SimpleShardList(str(tar_path)),
        wds.split_by_worker,
        wds.tarfile_to_samples(handler=log_and_continue),
        wds.map(decode_sample),
        wds.select(lambda x: x is not None),
        wds.batched(batch_size, collation_fn=dict_collate, partial=True),
    )
    loader = DataLoader(dataset, num_workers=4, batch_size=None)

    all_scores = {}

    for batch in tqdm(loader, desc=f"Stage1 {tar_stem}"):
        keys = batch["__key__"]
        images_pil = batch["image"]
        orig_texts = batch["txt"]
        json_list = batch["json"]
        n = len(keys)

        cleaned = [_clean(t) for t in orig_texts]

        # Resolve synthetic captions
        synthetic_texts = None
        short_synthetic_texts = None
        if compute_synthetic:
            synthetic_texts, short_synthetic_texts = _resolve_synthetic_captions(
                keys, orig_texts, json_list, caption_df
            )

        with torch.no_grad():
            scores_batch = _compute_reward_scores(
                images_pil, cleaned, synthetic_texts, short_synthetic_texts,
                compute_clip_score=compute_clip_score,
                compute_aesthetic_score=compute_aesthetic_score,
                compute_image_reward=compute_image_reward,
                compute_pickscore=compute_pickscore,
                compute_hpsv2=compute_hpsv2,
                compute_sciscore=compute_sciscore,
            )

        # Store per-sample scores
        for i in range(n):
            sample_scores = {}
            for score_name, arr in scores_batch.items():
                if arr is not None:
                    sample_scores[score_name] = float(arr[i])
            # Also store synthetic caption text in the scores CSV
            if synthetic_texts and i < len(synthetic_texts):
                sample_scores["synthetic_caption"] = synthetic_texts[i]
            if short_synthetic_texts and i < len(short_synthetic_texts):
                sample_scores["short_synthetic_caption"] = short_synthetic_texts[i]
            all_scores[keys[i]] = sample_scores

    write_csv_scores(csv_file, all_scores)
    logger.info("Stage 1 complete for %s -> %s", tar_path, csv_file)


def _resolve_synthetic_captions(keys, orig_texts, json_list, caption_df):
    """Get synthetic captions from caption_df or from sample JSON."""
    synthetic_texts = []
    short_synthetic_texts = []
    n = len(keys)

    if caption_df is not None:
        for i, key in enumerate(keys):
            img_filename = os.path.splitext(key.split("/")[-1])[0]
            if img_filename in caption_df.index:
                row = caption_df.loc[img_filename]
                # Last column = long caption, first column = short caption
                long_cap = row.iloc[-1] if isinstance(row.iloc[-1], str) else orig_texts[i]
                short_cap = row.iloc[0] if isinstance(row.iloc[0], str) else orig_texts[i]
                synthetic_texts.append(long_cap)
                short_synthetic_texts.append(short_cap)
            else:
                synthetic_texts.append(orig_texts[i])
                short_synthetic_texts.append(orig_texts[i])
    else:
        for i in range(n):
            j = json_list[i]
            syn = j.get("synthetic_caption", orig_texts[i])
            syn_short = j.get("short_synthetic_caption", j.get("caption", orig_texts[i]))
            if not isinstance(syn, str):
                syn = orig_texts[i]
            if not isinstance(syn_short, str):
                syn_short = syn
            synthetic_texts.append(syn)
            short_synthetic_texts.append(syn_short)

    synthetic_texts = [_clean(t) for t in synthetic_texts]
    short_synthetic_texts = [_clean(t) for t in short_synthetic_texts]
    return synthetic_texts, short_synthetic_texts


def _compute_reward_scores(
    images_pil, texts, synthetic_texts, short_synthetic_texts,
    compute_clip_score=True,
    compute_aesthetic_score=True,
    compute_image_reward=True,
    compute_pickscore=True,
    compute_hpsv2=True,
    compute_sciscore=True,
):
    """Compute all reward scores (except VQA) for a batch. Returns dict of arrays."""
    scores = {}

    # -- CLIP score --
    if compute_clip_score and clip_model is not None:
        clip_img = clip_image_transform(images_pil, return_tensors="pt")
        clip_img = {k: v.to(device) for k, v in clip_img.items()}

        clip_tok = clip_tokenizer(
            texts, truncation=True, padding="longest", max_length=1024, return_tensors="pt"
        )
        clip_tok = {k: v.to(device) for k, v in clip_tok.items()}
        out = clip_model(**clip_img, **clip_tok, output_hidden_states=True)
        scores["clip_score"] = torch.diag(out.logits_per_image).float().detach().cpu().numpy() / 100.0

        if synthetic_texts:
            syn_tok = clip_tokenizer(
                synthetic_texts, truncation=True, padding="longest", max_length=1024, return_tensors="pt"
            )
            syn_tok = {k: v.to(device) for k, v in syn_tok.items()}
            syn_out = clip_model(**clip_img, **syn_tok, output_hidden_states=True)
            scores["synthetic_clip_score"] = torch.diag(syn_out.logits_per_image).float().detach().cpu().numpy() / 100.0

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        # -- Aesthetic score --
        if compute_aesthetic_score and aesthetic_scorer is not None:
            scores["aesthetic_score"] = (aesthetic_scorer(images_pil).cpu().numpy() / 10.0)
            # Aesthetic is image-only, same for synthetic
            scores["synthetic_aesthetic_score"] = scores["aesthetic_score"].copy()

        # -- Image Reward --
        if compute_image_reward and image_reward_model is not None:
            scores["image_reward_score"] = (
                np.array(image_reward_model.score(texts, images_pil)) + 2
            ) / 4
            if short_synthetic_texts:
                scores["synthetic_image_reward_score"] = (
                    np.array(image_reward_model.score(short_synthetic_texts, images_pil)) + 2
                ) / 4

        # -- PickScore --
        if compute_pickscore and pickscore_model is not None:
            ps_data = pickscore_processor(
                images=images_pil, text=texts, padding=True, truncation=True,
                max_length=77, return_tensors="pt"
            )
            ps_data = {k: v.to(device) for k, v in ps_data.items()}
            scores["pick_a_score_score"] = (
                torch.diag(pickscore_model(**ps_data).logits_per_image) / 100.0
            ).float().detach().cpu().numpy()

            if short_synthetic_texts:
                syn_ps = pickscore_processor(
                    images=images_pil, text=short_synthetic_texts, padding=True,
                    truncation=True, max_length=77, return_tensors="pt"
                )
                syn_ps = {k: v.to(device) for k, v in syn_ps.items()}
                scores["synthetic_pick_a_score_score"] = (
                    torch.diag(pickscore_model(**syn_ps).logits_per_image) / 100.0
                ).float().detach().cpu().numpy()

        # -- HPSv2 --
        if compute_hpsv2 and hpsv2 is not None:
            scores["hpsv2_score"] = hpsv2.score(images_pil, texts)
            if short_synthetic_texts:
                scores["synthetic_hpsv2_score"] = hpsv2.score(images_pil, short_synthetic_texts)

        # -- SciScore --
        if compute_sciscore and sciscore_model is not None:
            sci_img = sciscore_processor(
                images=images_pil, padding=True, truncation=True, max_length=77, return_tensors="pt"
            )
            sci_img = {k: v.to(device) for k, v in sci_img.items()}
            sci_txt = sciscore_processor(
                text=texts, padding=True, truncation=True, max_length=77, return_tensors="pt"
            )
            sci_txt = {k: v.to(device) for k, v in sci_txt.items()}

            img_embs = sciscore_model.get_image_features(**sci_img)
            img_embs = img_embs / torch.norm(img_embs, dim=-1, keepdim=True)
            txt_embs = sciscore_model.get_text_features(**sci_txt)
            txt_embs = txt_embs / torch.norm(txt_embs, dim=-1, keepdim=True)
            logits = sciscore_model.logit_scale.exp() * (txt_embs @ img_embs.T)
            scores["sciscore_score"] = torch.diag(logits).float().detach().cpu().numpy() / 100.0

            if short_synthetic_texts:
                syn_sci_txt = sciscore_processor(
                    text=short_synthetic_texts, padding=True, truncation=True,
                    max_length=77, return_tensors="pt"
                )
                syn_sci_txt = {k: v.to(device) for k, v in syn_sci_txt.items()}
                syn_txt_embs = sciscore_model.get_text_features(**syn_sci_txt)
                syn_txt_embs = syn_txt_embs / torch.norm(syn_txt_embs, dim=-1, keepdim=True)
                syn_logits = sciscore_model.logit_scale.exp() * (syn_txt_embs @ img_embs.T)
                scores["synthetic_sciscore_score"] = torch.diag(syn_logits).float().detach().cpu().numpy() / 100.0

    return scores


# ---------------------------------------------------------------------------
# Stage 2: VQA Score
# ---------------------------------------------------------------------------
def stage_vqa(
    tar_path,
    csv_dir,
    batch_size=64,
    caption_df=None,
    compute_synthetic=True,
):
    """Compute VQAScore for original + synthetic captions. Update the CSV."""
    csv_file = csv_path_for_tar(csv_dir, tar_path)
    tar_stem = Path(tar_path).stem

    dataset = wds.DataPipeline(
        wds.SimpleShardList(str(tar_path)),
        wds.split_by_worker,
        wds.tarfile_to_samples(handler=log_and_continue),
        wds.map(decode_sample),
        wds.select(lambda x: x is not None),
        wds.batched(batch_size, collation_fn=dict_collate, partial=True),
    )
    loader = DataLoader(dataset, num_workers=4, batch_size=None)

    new_scores = {}

    for batch in tqdm(loader, desc=f"Stage2-VQA {tar_stem}"):
        keys = batch["__key__"]
        images_pil = batch["image"]
        orig_texts = batch["txt"]
        json_list = batch["json"]
        n = len(keys)

        cleaned = [_clean(t) for t in orig_texts]

        synthetic_texts = None
        if compute_synthetic:
            synthetic_texts, _ = _resolve_synthetic_captions(
                keys, orig_texts, json_list, caption_df
            )

        with torch.no_grad():
            vqa_scores = vqa_processor(images_pil, cleaned).cpu().numpy()
            for i in range(n):
                new_scores.setdefault(keys[i], {})["vqa_score"] = float(vqa_scores[i])

            if synthetic_texts:
                syn_vqa = vqa_processor(images_pil, synthetic_texts).cpu().numpy()
                for i in range(n):
                    new_scores[keys[i]]["synthetic_vqa_score"] = float(syn_vqa[i])

    update_csv_scores(csv_file, new_scores)
    logger.info("Stage 2 (VQA) complete for %s -> %s", tar_path, csv_file)


# ---------------------------------------------------------------------------
# Stage 3: VAE encoding + T5 text embeddings + output tar
# ---------------------------------------------------------------------------
def stage_vae(
    tar_path,
    csv_dir,
    dest_dir,
    batch_size=64,
    caption_df=None,
    compute_vae_256=True,
    compute_vae_512=True,
    compute_text_embeddings=True,
    compute_synthetic=True,
    text_encoder_model="google/flan-t5-xl",
    text_max_length=2048,
    embeddings_npy_name="flan_t5_xl_embeddings",
):
    """Encode images with SDXL VAE, compute T5 embeddings, and write the final
    output tar.

    Reads scores from the CSV (stages 1+2) and the original tar.  Computes VAE
    latents and T5 text embeddings for both original and synthetic captions
    inline, then writes an enriched output tar to dest_dir.
    """
    csv_file = csv_path_for_tar(csv_dir, tar_path)
    tar_stem = Path(tar_path).stem
    dest_path = dest_dir / Path(tar_path).name

    # Load scores from stages 1+2
    scores_dict = read_csv_scores(csv_file)

    # Prepare a sample preprocessor that adds VAE image tensors
    def preprocess(sample):
        decoded = decode_sample(sample)
        if decoded is None:
            return None
        if compute_vae_256:
            decoded["vae_image_256"] = vae_image_transforms_256(decoded["image"])
        if compute_vae_512:
            decoded["vae_image_512"] = vae_image_transforms_512(decoded["image"])
        return decoded

    dataset = wds.DataPipeline(
        wds.SimpleShardList(str(tar_path)),
        wds.split_by_worker,
        wds.tarfile_to_samples(handler=log_and_continue),
        wds.map(preprocess),
        wds.select(lambda x: x is not None),
        wds.batched(batch_size, collation_fn=dict_collate, partial=True),
    )
    loader = DataLoader(dataset, num_workers=4, batch_size=None)

    with wds.TarWriter(str(dest_path)) as sink:
        for batch in tqdm(loader, desc=f"Stage3 {tar_stem}"):
            keys = batch["__key__"]
            orig_texts = batch["txt"]
            json_list = batch["json"]
            orig_bytes = batch["orig_image_bytes"]
            n = len(keys)

            cleaned = [_clean(t) for t in orig_texts]

            # Resolve synthetic captions for T5 encoding
            synthetic_texts = None
            if compute_synthetic:
                synthetic_texts, _ = _resolve_synthetic_captions(
                    keys, orig_texts, json_list, caption_df
                )

            vae_256_mean = vae_256_std = vae_512_mean = vae_512_std = None
            text_embs = synthetic_text_embs = None
            text_emb_lengths = synthetic_text_emb_lengths = None

            with torch.no_grad():
                # -- VAE encoding --
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    if compute_vae_256:
                        vae_in = batch["vae_image_256"].to(device)
                        dist = vae_model.encode(vae_in).latent_dist
                        vae_256_mean = dist.mean.half().cpu().numpy()
                        vae_256_std = dist.std.half().cpu().numpy()

                    if compute_vae_512:
                        vae_in = batch["vae_image_512"].to(device)
                        dist = vae_model.encode(vae_in).latent_dist
                        vae_512_mean = dist.mean.half().cpu().numpy()
                        vae_512_std = dist.std.half().cpu().numpy()

                # -- T5 text embeddings --
                if compute_text_embeddings and text_model is not None:
                    # Original captions
                    tokenized = text_tokenizer(
                        cleaned, return_tensors="pt", padding="longest",
                        max_length=text_max_length, truncation=True,
                    )
                    tokenized = {k: v.to(device) for k, v in tokenized.items()}
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        text_embs = text_model(**tokenized).last_hidden_state.detach().half().cpu().numpy()
                    text_emb_lengths = [
                        tokenized["attention_mask"][i].sum().item() for i in range(n)
                    ]

                    # Synthetic captions
                    if synthetic_texts:
                        syn_cleaned = [_clean(t) for t in synthetic_texts]
                        syn_tokenized = text_tokenizer(
                            syn_cleaned, return_tensors="pt", padding="longest",
                            max_length=text_max_length, truncation=True,
                        )
                        syn_tokenized = {k: v.to(device) for k, v in syn_tokenized.items()}
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            synthetic_text_embs = text_model(**syn_tokenized).last_hidden_state.detach().half().cpu().numpy()
                        synthetic_text_emb_lengths = [
                            syn_tokenized["attention_mask"][i].sum().item() for i in range(n)
                        ]

            # -- Write samples --
            for i in range(n):
                key = keys[i]
                meta = json_list[i]

                # Merge CSV scores into json metadata
                if key in scores_dict:
                    for score_key, val in scores_dict[key].items():
                        if val is not None:
                            # Keep caption strings as strings, scores as floats
                            if "caption" in score_key:
                                meta[score_key] = str(val) if not isinstance(val, str) else val
                            else:
                                try:
                                    meta[score_key] = float(val)
                                except (ValueError, TypeError):
                                    meta[score_key] = val

                sample = {
                    "__key__": key,
                    "jpg": orig_bytes[i],
                    "txt": orig_texts[i],
                    "json": meta,
                }

                # VAE embeddings
                if compute_vae_256 and vae_256_mean is not None:
                    sample["vae_embeddings_mean_256.npy"] = vae_256_mean[i]
                    sample["vae_embeddings_std_256.npy"] = vae_256_std[i]
                if compute_vae_512 and vae_512_mean is not None:
                    sample["vae_embeddings_mean_512.npy"] = vae_512_mean[i]
                    sample["vae_embeddings_std_512.npy"] = vae_512_std[i]

                # T5 embeddings (trimmed to actual token length)
                if text_embs is not None:
                    sample[f"{embeddings_npy_name}.npy"] = text_embs[i][:text_emb_lengths[i]]
                if synthetic_text_embs is not None:
                    sample[f"synthetic_{embeddings_npy_name}.npy"] = synthetic_text_embs[i][:synthetic_text_emb_lengths[i]]

                sink.write(sample)

    logger.info("Stage 3 (VAE+T5) complete for %s -> %s", tar_path, dest_path)


# ---------------------------------------------------------------------------
# Shard selection helpers
# ---------------------------------------------------------------------------
def resolve_shards(src, shard_id=None, shard_ids=None, shard_range=None):
    """Return list of tar paths to process based on selection args."""
    all_shards = sorted(Path(src).glob("*.tar"))
    if not all_shards:
        raise FileNotFoundError(f"No .tar files found in {src}")

    if shard_id is not None:
        idx = int(shard_id)
        if 0 <= idx < len(all_shards):
            return [all_shards[idx]]
        raise ValueError(f"Shard ID {idx} out of range (0-{len(all_shards)-1})")

    if shard_ids is not None:
        result = []
        for s in shard_ids.split(","):
            idx = int(s.strip())
            if 0 <= idx < len(all_shards):
                result.append(all_shards[idx])
            else:
                logger.warning("Shard ID %d out of range, skipping", idx)
        return result

    if shard_range is not None:
        start, end = map(int, shard_range.split("-"))
        if start < 0 or end >= len(all_shards) or start > end:
            raise ValueError(f"Invalid range {start}-{end} (max {len(all_shards)-1})")
        return all_shards[start:end + 1]

    return all_shards


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Multi-stage preprocessing for raw webdataset tars",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--stage", required=True, choices=["rewards", "vqa", "vae", "all"],
                        help="Which stage to run")
    parser.add_argument("--src", required=True, help="Directory with source .tar files")
    parser.add_argument("--dest", help="Output directory for enriched tars (stage vae/all)")
    parser.add_argument("--csv_dir", help="Directory for per-tar CSV score files (default: src/scores)")

    # Shard selection
    parser.add_argument("--shard_id", help="Single shard index to process")
    parser.add_argument("--shard_ids", help="Comma-separated shard indices")
    parser.add_argument("--shard_range", help="Range start-end (inclusive)")

    # Synthetic captions
    parser.add_argument("--synthetic_captions", help="TSV file with synthetic captions")
    parser.add_argument("--no_synthetic", action="store_true", help="Skip synthetic caption scoring")

    # Model / embedding config
    parser.add_argument("--text_encoder_model", default="google/flan-t5-xl")
    parser.add_argument("--text_max_length", type=int, default=2048)
    parser.add_argument("--embeddings_npy_name", default="flan_t5_xl_embeddings",
                        help="Base name for text embedding files (no .npy)")
    parser.add_argument("--batch_size", type=int, default=64)

    # Disable individual rewards (stage 1)
    parser.add_argument("--no_clip_score", action="store_true")
    parser.add_argument("--no_aesthetic_score", action="store_true")
    parser.add_argument("--no_image_reward", action="store_true")
    parser.add_argument("--no_pickscore", action="store_true")
    parser.add_argument("--no_hpsv2", action="store_true")
    parser.add_argument("--no_sciscore", action="store_true")
    parser.add_argument("--no_text_embeddings", action="store_true")

    # VAE config (stage 3)
    parser.add_argument("--no_vae_256", action="store_true")
    parser.add_argument("--no_vae_512", action="store_true")

    args = parser.parse_args()

    src = Path(args.src)
    csv_dir = Path(args.csv_dir) if args.csv_dir else src / "scores"
    csv_dir.mkdir(parents=True, exist_ok=True)

    shards = resolve_shards(
        src, shard_id=args.shard_id, shard_ids=args.shard_ids,
        shard_range=args.shard_range,
    )
    logger.info("Processing %d shards (stage=%s)", len(shards), args.stage)

    compute_synthetic = not args.no_synthetic

    # Load synthetic captions if provided
    caption_df = None
    if args.synthetic_captions and compute_synthetic:
        import pandas as pd
        logger.info("Loading synthetic captions from %s", args.synthetic_captions)
        caption_df = pd.read_csv(args.synthetic_captions, sep="\t", header=None)
        caption_df[0] = caption_df[0].apply(lambda x: os.path.splitext(x)[0])
        caption_df.set_index(0, inplace=True)
        logger.info("Loaded %d synthetic captions", len(caption_df))

    # ---- Stage: rewards ----
    if args.stage in ("rewards", "all"):
        c_clip = not args.no_clip_score
        c_aes = not args.no_aesthetic_score
        c_ir = not args.no_image_reward
        c_ps = not args.no_pickscore
        c_hps = not args.no_hpsv2
        c_sci = not args.no_sciscore

        # Load models
        load_reward_models(
            load_clip=c_clip,
            load_aesthetic_score=c_aes,
            load_image_reward=c_ir,
            load_pickscore=c_ps,
            load_hpsv2=c_hps,
            load_sciscore=c_sci,
        )

        for shard in tqdm(shards, desc="Stage rewards"):
            stage_rewards(
                shard, csv_dir,
                batch_size=args.batch_size,
                caption_df=caption_df,
                compute_clip_score=c_clip,
                compute_aesthetic_score=c_aes,
                compute_image_reward=c_ir,
                compute_pickscore=c_ps,
                compute_hpsv2=c_hps,
                compute_sciscore=c_sci,
                compute_synthetic=compute_synthetic,
            )

        # Free GPU memory before next stage
        if args.stage == "all":
            _unload_reward_models()

    # ---- Stage: vqa ----
    if args.stage in ("vqa", "all"):
        load_vqa_model()

        for shard in tqdm(shards, desc="Stage VQA"):
            stage_vqa(
                shard, csv_dir,
                batch_size=args.batch_size,
                caption_df=caption_df,
                compute_synthetic=compute_synthetic,
            )

        if args.stage == "all":
            _unload_vqa_model()

    # ---- Stage: vae (+ T5) ----
    if args.stage in ("vae", "all"):
        if args.dest is None:
            parser.error("--dest is required for stage vae / all")
        dest_dir = Path(args.dest)
        dest_dir.mkdir(parents=True, exist_ok=True)

        c_te = not args.no_text_embeddings
        load_vae_model()
        if c_te:
            load_text_model(args.text_encoder_model)

        for shard in tqdm(shards, desc="Stage VAE+T5"):
            stage_vae(
                shard, csv_dir, dest_dir,
                batch_size=args.batch_size,
                caption_df=caption_df,
                compute_vae_256=not args.no_vae_256,
                compute_vae_512=not args.no_vae_512,
                compute_text_embeddings=c_te,
                compute_synthetic=compute_synthetic,
                text_encoder_model=args.text_encoder_model,
                text_max_length=args.text_max_length,
                embeddings_npy_name=args.embeddings_npy_name,
            )

    logger.info("All done.")


def _unload_reward_models():
    """Free GPU memory from reward models."""
    global clip_model, clip_tokenizer, clip_image_transform
    global aesthetic_scorer, image_reward_model
    global pickscore_processor, pickscore_model
    global hpsv2, sciscore_processor, sciscore_model
    clip_model = clip_tokenizer = clip_image_transform = None
    aesthetic_scorer = image_reward_model = None
    pickscore_processor = pickscore_model = None
    hpsv2 = sciscore_processor = sciscore_model = None
    torch.cuda.empty_cache()


def _unload_vqa_model():
    global vqa_processor
    vqa_processor = None
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
