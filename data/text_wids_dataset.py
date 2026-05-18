"""Text-conditional dataset using WIDS (Web Indexed Dataset Shards).

Replaces the webdataset-based TextWebDataset with a random-access Dataset
backed by ShardListDataset from cad/data/wids/.
"""

import glob
import json
import logging
import math
import os
import random
import tarfile as _tarfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from lightning_fabric.utilities.rank_zero import _get_rank
from torch.utils.data import Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from miro.data.wids import ShardListDataset
from miro.utils.misc import print0


def _count_samples_in_tar(tar_path):
    """Count the number of webdataset samples in a tar by reading member names.

    A sample is a group of files sharing the same key (basename before first dot).
    This is much faster than IndexedTarSamples since it only reads the tar index,
    not the file contents.

    Returns:
        tuple: (tar_path, nsamples)
    """
    keys = set()
    with _tarfile.open(tar_path, "r") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            name = member.name
            if "." in name:
                key = name[: name.index(".")]
                keys.add(key)
    return tar_path, len(keys)


def _build_wids_index(root_dir, num_workers=16):
    """Build or load a wids-index.json for a directory of tar files.

    Uses parallel workers to count samples by reading tar member names
    (no mmap/full indexing). Only rank 0 writes the index.

    Returns:
        list[dict]: shard list entries with "url" and "nsamples" keys.
    """
    index_path = os.path.join(root_dir, "wids-index.json")
    if os.path.exists(index_path):
        print0(f"Loading WIDS index from {index_path}")
        with open(index_path, "r") as f:
            data = json.load(f)
        return data["shardlist"]

    print0(f"Building WIDS index for {root_dir}...")
    tar_files = sorted(glob.glob(os.path.join(root_dir, "*.tar")))
    if not tar_files:
        raise FileNotFoundError(f"No .tar files found in {root_dir}")

    # Count samples in parallel
    effective_workers = min(num_workers, len(tar_files))
    if effective_workers > 1:
        with Pool(effective_workers) as pool:
            results = list(tqdm(
                pool.imap(_count_samples_in_tar, tar_files),
                total=len(tar_files),
                desc=f"Indexing {root_dir}",
                disable=_get_rank() != 0,
            ))
    else:
        results = [_count_samples_in_tar(f) for f in tqdm(
            tar_files, desc=f"Indexing {root_dir}", disable=_get_rank() != 0,
        )]

    shardlist = [{"url": path, "nsamples": n} for path, n in results]

    index_data = {"wids_version": 1, "shardlist": shardlist}
    with open(index_path, "w") as f:
        json.dump(index_data, f)
    total = sum(s["nsamples"] for s in shardlist)
    print0(f"Saved WIDS index to {index_path} ({len(shardlist)} shards, {total} samples)")
    return shardlist


def _compute_bins(values, num_bins, strategy, refine_last_n_bins=0, refine_factor=1):
    """Compute bin edges from a tensor of values.

    Supports 'uniform', 'quantile', and 'refined_quantile' strategies.

    Returns:
        torch.Tensor of bin edges, or None if not enough data.
    """
    unique_values = torch.unique(values)

    if strategy == "refined_quantile" and refine_last_n_bins > 0:
        actual_num_bins = (num_bins - refine_last_n_bins) + (refine_last_n_bins * refine_factor)
    else:
        actual_num_bins = num_bins

    if strategy == "uniform":
        bins = torch.linspace(unique_values.min().item(), unique_values.max().item(), num_bins + 1)
    elif strategy == "quantile":
        if len(unique_values) < num_bins + 1:
            print0(f"Warning: Not enough unique values ({len(unique_values)}) for {num_bins} bins. Falling back to uniform.")
            bins = torch.linspace(unique_values.min().item(), unique_values.max().item(), num_bins + 1)
        else:
            try:
                bins = torch.quantile(unique_values, q=torch.linspace(0, 1, num_bins + 1))
            except RuntimeError as e:
                print0(f"Warning: torch.quantile failed: {e}. Falling back to uniform.")
                bins = torch.linspace(unique_values.min().item(), unique_values.max().item(), num_bins + 1)
    elif strategy == "refined_quantile":
        total_bins_needed = actual_num_bins + 1
        if len(unique_values) < total_bins_needed:
            print0(f"Warning: Not enough unique values for {actual_num_bins} refined bins. Falling back to uniform.")
            bins = torch.linspace(unique_values.min().item(), unique_values.max().item(), total_bins_needed)
        else:
            try:
                quantile_positions = [i / num_bins for i in range(num_bins - refine_last_n_bins + 1)]
                if refine_last_n_bins > 0:
                    start_q = (num_bins - refine_last_n_bins) / num_bins
                    range_q = 1.0 - start_q
                    num_refined = refine_last_n_bins * refine_factor
                    for i in range(1, num_refined + 1):
                        quantile_positions.append(start_q + i * range_q / num_refined)
                bins = torch.quantile(unique_values, q=torch.tensor(quantile_positions, dtype=torch.float32))
                print0(f"Created {actual_num_bins} refined bins (last {refine_last_n_bins} subdivided by {refine_factor}x)")
            except RuntimeError as e:
                print0(f"Warning: refined quantile failed: {e}. Falling back to uniform.")
                bins = torch.linspace(unique_values.min().item(), unique_values.max().item(), total_bins_needed)
    else:
        raise ValueError(f"Invalid bin_strategy: {strategy}. Must be 'quantile', 'uniform', or 'refined_quantile'.")

    bins[0] = 0.0
    bins[-1] = 1.0
    bins = torch.unique(bins)

    if len(bins) < 2:
        return None
    return bins


def compute_coherence_bins(
    dataset,
    coherence_scores,
    num_bins,
    bin_strategy,
    refine_last_n_bins,
    refine_factor,
    max_samples,
    checkpoint_path,
    synthetic_embedding_prob=0.0,
    text_embedding_name=None,
):
    """Compute coherence score bins, with distributed support and caching.

    Args:
        dataset: ShardListDataset (raw WIDS dataset) for reading .json metadata.
        coherence_scores: list of score names to bin.
        num_bins: number of bins.
        bin_strategy: 'uniform', 'quantile', or 'refined_quantile'.
        refine_last_n_bins: for refined_quantile strategy.
        refine_factor: for refined_quantile strategy.
        max_samples: max samples for bin estimation.
        checkpoint_path: Path to cache file.
        synthetic_embedding_prob: probability of synthetic data.
        text_embedding_name: name of text embedding (for synthetic selection).

    Returns:
        dict[str, torch.Tensor]: bin edges per score name.
    """
    rank = _get_rank()
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    is_distributed = world_size > 1

    # --- Check cache ---
    bins = {}
    cache_info = {"found": False, "bins_data": None}

    if rank == 0 or not is_distributed:
        if checkpoint_path.exists():
            print0(f"Loading coherence bins from {checkpoint_path}")
            try:
                with open(checkpoint_path, "r") as f:
                    bins_data = json.load(f)
                if set(bins_data.keys()) == set(coherence_scores):
                    cache_info["found"] = True
                    cache_info["bins_data"] = bins_data
                    print0(f"Loaded bins from cache: {list(bins_data.keys())}")
                else:
                    print0("Cached bins keys mismatch. Recalculating.")
            except (json.JSONDecodeError, Exception) as e:
                print0(f"Error loading {checkpoint_path}: {e}. Recomputing.")

    if is_distributed:
        broadcast_list = [cache_info]
        torch.distributed.broadcast_object_list(broadcast_list, src=0)
        cache_info = broadcast_list[0]

    if cache_info["found"]:
        return {k: torch.tensor(v, dtype=torch.float32) for k, v in cache_info["bins_data"].items()}

    # --- Gather values ---
    # Fast path: temporarily swap the dataset's transformations so we only
    # decode the .json metadata. _selective_decode otherwise pulls all needed
    # .npy embeddings (synthetic_flan_t5_xl_embeddings, vae_embeddings_*,
    # ...) which dominates the per-sample cost on Lustre and made bin
    # estimation effectively unusable (5-6 s/sample). The JSON members are
    # small and cheap to extract.
    def _json_only_decode(sample):
        out = {}
        for key, stream in sample.items():
            if key.startswith("__"):
                out[key] = stream
            elif key.endswith(".json"):
                data = stream.read() if hasattr(stream, "read") else stream
                out[key] = json.loads(data)
        return out

    _orig_transformations = getattr(dataset, "transformations", None)
    if _orig_transformations is not None:
        dataset.transformations = [_json_only_decode]

    try:
        dataset_len = len(dataset)
        if is_distributed:
            samples_per_rank = max(1, max_samples // world_size)
            # Each rank samples a different portion
            rng = random.Random(42 + rank)
            indices = rng.sample(range(dataset_len), min(samples_per_rank, dataset_len))
        else:
            rng = random.Random(42)
            indices = rng.sample(range(dataset_len), min(max_samples, dataset_len))

        local_values = {score: [] for score in coherence_scores}
        for idx in tqdm(indices, desc=f"Gathering coherence values (rank {rank})", disable=rank != 0):
            try:
                sample = dataset[idx]
                metadata = sample.get(".json", {})
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)

                # Handle synthetic selection for bin estimation
                if synthetic_embedding_prob > 0 and random.random() < synthetic_embedding_prob:
                    for score_name in coherence_scores:
                        synthetic_key = f"synthetic_{score_name}_score"
                        regular_key = f"{score_name}_score"
                        if synthetic_key in metadata and regular_key in metadata:
                            metadata[regular_key] = metadata[synthetic_key]

                for score_name in coherence_scores:
                    if score_name in metadata:
                        local_values[score_name].append(metadata[score_name])
            except Exception as e:
                logging.warning(f"Error reading sample {idx} for binning: {e}")
                continue
    finally:
        if _orig_transformations is not None:
            dataset.transformations = _orig_transformations

    # --- Reduce and compute bins ---
    if is_distributed:
        gathered_list = [None] * world_size
        torch.distributed.gather_object(local_values, gathered_list if rank == 0 else None, dst=0)

        computed_bins_data = None
        if rank == 0:
            combined = {score: [] for score in coherence_scores}
            for rank_values in gathered_list:
                if rank_values:
                    for score_name, values in rank_values.items():
                        combined[score_name].extend(values)

            for score_name, values in combined.items():
                if not values:
                    print0(f"Warning: No values for '{score_name}'. Skipping.")
                    continue
                result = _compute_bins(
                    torch.tensor(values, dtype=torch.float32),
                    num_bins, bin_strategy, refine_last_n_bins, refine_factor,
                )
                if result is not None:
                    bins[score_name] = result
                    print0(f"Computed bins for {score_name}: {result}")

            if bins:
                computed_bins_data = {k: v.tolist() for k, v in bins.items()}
                print0(f"Saving coherence bins to {checkpoint_path}")
                try:
                    with open(checkpoint_path, "w") as f:
                        json.dump(computed_bins_data, f)
                except Exception as e:
                    print0(f"Error saving {checkpoint_path}: {e}")

        broadcast_bins = [computed_bins_data]
        torch.distributed.broadcast_object_list(broadcast_bins, src=0)

        if rank != 0:
            received = broadcast_bins[0]
            if received is not None:
                bins = {k: torch.tensor(v, dtype=torch.float32) for k, v in received.items()}
            else:
                print0(f"Rank {rank}: Bin computation failed on rank 0.")

        torch.distributed.barrier()
    else:
        for score_name, values in local_values.items():
            if not values:
                print0(f"Warning: No values for '{score_name}'. Skipping.")
                continue
            result = _compute_bins(
                torch.tensor(values, dtype=torch.float32),
                num_bins, bin_strategy, refine_last_n_bins, refine_factor,
            )
            if result is not None:
                bins[score_name] = result
                print0(f"Computed bins for {score_name}: {result}")

        if bins:
            print0(f"Saving coherence bins to {checkpoint_path}")
            bins_to_save = {k: v.tolist() for k, v in bins.items()}
            try:
                with open(checkpoint_path, "w") as f:
                    json.dump(bins_to_save, f)
            except Exception as e:
                print0(f"Error saving {checkpoint_path}: {e}")

    return bins


class ShardExclusiveSampler(DistributedSampler):
    """Sampler that assigns exclusive shard ranges to each DataLoader worker.

    Organizes the index sequence so that PyTorch DataLoader's round-robin
    batch dispatch gives each worker indices from non-overlapping shards.
    This minimizes the number of tar files each worker opens, maximizing
    LRU cache efficiency and avoiding Lustre file contention.

    The output interleaves worker blocks:
        [worker_0 batch] [worker_1 batch] ... [worker_N batch] [worker_0 batch] ...
    Since DataLoader dispatches batch i to worker (i % num_workers), each
    worker only sees its own shard group.

    Inherits from DistributedSampler so PyTorch Lightning recognizes it
    and calls set_epoch() automatically.
    """

    def __init__(
        self,
        dataset,
        num_workers,
        batch_size,
        *,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
    ):
        import torch.distributed as dist

        if not dist.is_initialized():
            num_replicas = 1
            rank = 0
        else:
            num_replicas = num_replicas or dist.get_world_size()
            rank = rank if rank is not None else dist.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = True
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.num_workers = max(1, num_workers)
        self.batch_size = batch_size

        # Get shard boundaries from the underlying WIDS dataset
        wids = dataset.dataset if hasattr(dataset, "dataset") else dataset
        self.shard_lengths = list(wids.lengths)
        self.cum_lengths = np.cumsum([0] + self.shard_lengths)

        # Partition across DDP ranks (contiguous shard ranges)
        shards_per_rank = len(self.shard_lengths) // num_replicas
        shard_start = rank * shards_per_rank
        shard_end = (
            (rank + 1) * shards_per_rank
            if rank < num_replicas - 1
            else len(self.shard_lengths)
        )
        self.rank_shard_range = (shard_start, shard_end)

        # Compute total samples for this rank (raw count)
        self._raw_num_samples = sum(self.shard_lengths[shard_start:shard_end])

        # Compute global max batches across all ranks so __len__ is consistent
        global_max_batches = 0
        for r in range(num_replicas):
            r_s = r * (len(self.shard_lengths) // num_replicas)
            r_e = (
                (r + 1) * (len(self.shard_lengths) // num_replicas)
                if r < num_replicas - 1
                else len(self.shard_lengths)
            )
            r_total = sum(self.shard_lengths[r_s:r_e])
            r_batches = (r_total // self.num_workers) // batch_size
            if r_batches > global_max_batches:
                global_max_batches = r_batches
        # Total yielded = global_max_batches * num_workers * batch_size
        self._num_samples = global_max_batches * self.num_workers * batch_size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self._num_samples

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        shard_start, shard_end = self.rank_shard_range
        num_shards = shard_end - shard_start
        nw = self.num_workers

        # Shuffle shard order for this rank, then assign to workers.
        # The shuffle changes every epoch, so workers see different shards each epoch.
        rank_shards = list(range(shard_start, shard_end))
        rng.shuffle(rank_shards)

        # Assign shards to workers (contiguous groups from the shuffled list)
        shards_per_worker = num_shards // nw
        worker_shard_lists = []
        for w in range(nw):
            ws = w * shards_per_worker
            we = (w + 1) * shards_per_worker if w < nw - 1 else num_shards
            worker_shard_lists.append(rank_shards[ws:we])

        # Build per-worker index lists (shuffled within shards)
        worker_indices = []
        for shard_list in worker_shard_lists:
            indices = []
            if self.shuffle:
                rng.shuffle(shard_list)
            for s in shard_list:
                lo = int(self.cum_lengths[s])
                hi = int(self.cum_lengths[s + 1])
                shard_indices = list(range(lo, hi))
                if self.shuffle:
                    rng.shuffle(shard_indices)
                indices.extend(shard_indices)
            worker_indices.append(indices)

        # Interleave in batch-sized blocks so DataLoader round-robin
        # dispatches each block to the correct worker
        bs = self.batch_size
        result = []

        # Compute max_batches globally across ALL ranks to keep them in sync.
        # We can do this without communication since shard_lengths and the
        # partition scheme are deterministic and identical on every rank.
        global_max_batches = 0
        for r in range(self.num_replicas):
            r_shard_start = r * (len(self.shard_lengths) // self.num_replicas)
            r_shard_end = (
                (r + 1) * (len(self.shard_lengths) // self.num_replicas)
                if r < self.num_replicas - 1
                else len(self.shard_lengths)
            )
            r_total = sum(self.shard_lengths[r_shard_start:r_shard_end])
            # Each worker on rank r gets roughly r_total / nw samples
            r_per_worker = r_total // nw
            r_batches = r_per_worker // bs
            if r_batches > global_max_batches:
                global_max_batches = r_batches

        max_batches = global_max_batches

        for batch_idx in range(max_batches):
            for w in range(nw):
                start = batch_idx * bs
                end = start + bs
                if end <= len(worker_indices[w]):
                    result.extend(worker_indices[w][start:end])
                elif worker_indices[w]:
                    # Pad by wrapping to keep ranks synchronized
                    padded = []
                    for i in range(bs):
                        padded.append(worker_indices[w][i % len(worker_indices[w])])
                    result.extend(padded)

        return iter(result)


class ShuffledBatchLoader:
    """Sample-level shuffle buffer wrapping a DataLoader.

    Maintains a queue of individual samples drawn from all workers.
    Each output batch is assembled by randomly sampling from the queue,
    then the queue is refilled from the next worker batch. This gives
    full cross-worker mixing while I/O stays shard-exclusive.

    Args:
        dataloader: The underlying DataLoader (with shard-exclusive sampler).
        buffer_batches: Buffer size as a multiple of batch_size.
            Default is ``num_workers * 2`` batches worth of samples.
    """

    def __init__(self, dataloader, buffer_batches=None):
        self.dataloader = dataloader
        self.batch_size = dataloader.batch_size
        nw = max(getattr(dataloader, "num_workers", 1), 1)
        self.buffer_batches = buffer_batches or nw * 2
        self.buffer_size = self.buffer_batches * self.batch_size

    def _unbatch(self, batch):
        """Split a collated batch dict back into a list of sample dicts."""
        if not isinstance(batch, dict):
            return [batch]
        keys = list(batch.keys())
        bs = batch[keys[0]].shape[0] if hasattr(batch[keys[0]], "shape") else len(batch[keys[0]])
        return [{k: batch[k][i] for k in keys} for i in range(bs)]

    def _collate(self, samples):
        """Collate a list of sample dicts into a batch dict."""
        keys = list(samples[0].keys())
        batch = {}
        for k in keys:
            vals = [s[k] for s in samples]
            if isinstance(vals[0], torch.Tensor):
                batch[k] = torch.stack(vals)
            else:
                batch[k] = vals
        return batch

    def __iter__(self):
        buf = []
        it = iter(self.dataloader)

        # Fill the buffer
        while len(buf) < self.buffer_size:
            try:
                batch = next(it)
            except StopIteration:
                break
            buf.extend(self._unbatch(batch))

        # Yield batches by sampling from the buffer, refilling as we go
        while len(buf) >= self.batch_size:
            # Pick batch_size random indices from the buffer
            indices = random.sample(range(len(buf)), self.batch_size)
            indices.sort(reverse=True)
            samples = []
            for i in indices:
                samples.append(buf[i])
                buf[i] = buf[-1]
                buf.pop()

            yield self._collate(samples)

            # Refill from the DataLoader
            try:
                batch = next(it)
                buf.extend(self._unbatch(batch))
            except StopIteration:
                pass

        # Flush remaining full batches
        while len(buf) >= self.batch_size:
            random.shuffle(buf)
            yield self._collate(buf[: self.batch_size])
            buf = buf[self.batch_size :]

    def __len__(self):
        return len(self.dataloader)

    def __getattr__(self, name):
        return getattr(self.dataloader, name)


class TextWIDSDataset(Dataset):
    """Text-conditional dataset using WIDS for random-access tar reading.

    Drop-in replacement for TextWebDataset with the same constructor signature.
    Backed by ShardListDataset from cad/data/wids/.
    """

    def __init__(
        self,
        root,
        image_transforms=None,
        distributed=True,
        train=True,
        epoch=0,
        seed=3407,
        text_embedding_name=None,
        vae_embedding_name_mean=None,
        vae_embedding_name_std=None,
        repa_embedding_name=None,
        return_image=True,
        return_text=True,
        min_image_size=256,
        bin_coherence=False,
        num_bins=8,
        bin_strategy="quantile",
        refine_last_n_bins=0,
        refine_factor=1,
        clip_filter_threshold=0.0,
        aesthetic_threshold=0.0,
        shard_shuffle_size=2000,
        shard_shuffle_initial=500,
        sample_shuffle_size=5000,
        sample_shuffle_initial=1000,
        coherence_scores=None,
        max_samples_bin_estimation=100000,
        synthetic_embedding_prob=0.0,
        lru_size=10,
    ):
        super().__init__()
        self.image_transforms = image_transforms
        self.text_embedding_name = text_embedding_name
        self.vae_embedding_name_mean = vae_embedding_name_mean
        self.vae_embedding_name_std = vae_embedding_name_std
        self.repa_embedding_name = repa_embedding_name
        self.return_image = return_image
        self.return_text = return_text
        self.min_image_size = min_image_size
        self.clip_filter_threshold = clip_filter_threshold
        self.aesthetic_threshold = aesthetic_threshold
        self.coherence_scores = coherence_scores
        self.bin_coherence = bin_coherence
        self.num_bins = num_bins
        self.bin_strategy = bin_strategy
        self.refine_last_n_bins = refine_last_n_bins
        self.refine_factor = refine_factor
        self.synthetic_embedding_prob = synthetic_embedding_prob

        # Calculate actual number of bins after refinement
        if bin_strategy == "refined_quantile" and refine_last_n_bins > 0:
            self.actual_num_bins = (num_bins - refine_last_n_bins) + (refine_last_n_bins * refine_factor)
        else:
            self.actual_num_bins = num_bins

        # --- Parse root directories ---
        if " " in root:
            roots = root.split(" ")
            print0(f"Using multiple datasets: {roots}")
        else:
            roots = [root]

        # --- Build WIDS index for each root ---
        rank = _get_rank()
        is_distributed = torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1

        all_shards = []
        for r in roots:
            if rank == 0 or not is_distributed:
                shardlist = _build_wids_index(r)
            else:
                shardlist = None

            if is_distributed:
                broadcast_list = [shardlist]
                torch.distributed.broadcast_object_list(broadcast_list, src=0)
                shardlist = broadcast_list[0]

            all_shards.extend(shardlist)

        # --- Build set of keys we actually need (for selective decoding) ---
        self._needed_extensions = {".json"}
        if return_image:
            self._needed_extensions.update({".jpg", ".jpeg", ".png", ".webp"})
        if return_text:
            self._needed_extensions.add(".txt")
        if vae_embedding_name_mean:
            self._needed_extensions.add(f".{vae_embedding_name_mean}.npy")
        if vae_embedding_name_std:
            self._needed_extensions.add(f".{vae_embedding_name_std}.npy")
        if text_embedding_name:
            self._needed_extensions.add(f".{text_embedding_name}_embeddings.npy")
            if synthetic_embedding_prob > 0:
                self._needed_extensions.add(f".synthetic_{text_embedding_name}_embeddings.npy")
        if repa_embedding_name:
            self._needed_extensions.add(f".{repa_embedding_name}_embeddings.npy")

        # --- Create ShardListDataset with selective decoder ---
        self.dataset = ShardListDataset(
            all_shards,
            transformations=[self._selective_decode],
            localname=lambda x: x,
            lru_size=lru_size,
        )

        # --- Compute dataset size (with filtering estimation) ---
        self.num_samples = self._estimate_filtered_size(roots, all_shards)

        # --- Compute coherence bins ---
        self.bins = {}
        if coherence_scores and bin_coherence:
            root_for_cache = Path(roots[0])
            sorted_scores = sorted(coherence_scores)
            scores_str = "-".join(sorted_scores)
            if bin_strategy == "quantile":
                ckpt_name = f"coherence_bins_{num_bins}_bins_{scores_str}_synthetic_prob_{synthetic_embedding_prob}.json"
            elif bin_strategy == "refined_quantile":
                ckpt_name = f"coherence_bins_{num_bins}_bins_{scores_str}_strategy_{bin_strategy}_refine_{refine_last_n_bins}_factor_{refine_factor}_synthetic_prob_{synthetic_embedding_prob}.json"
            else:
                ckpt_name = f"coherence_bins_{num_bins}_bins_{scores_str}_strategy_{bin_strategy}_synthetic_prob_{synthetic_embedding_prob}.json"

            self.bins = compute_coherence_bins(
                self.dataset,
                coherence_scores,
                num_bins,
                bin_strategy,
                refine_last_n_bins,
                refine_factor,
                max_samples_bin_estimation,
                root_for_cache / ckpt_name,
                synthetic_embedding_prob,
                text_embedding_name,
            )

            if not self.bins:
                logging.warning("Coherence binning enabled but no bins computed. Disabling binning.")
                self.bin_coherence = False

        if is_distributed:
            torch.distributed.barrier()

    def _selective_decode(self, sample):
        """Decode only the fields we need, skipping expensive image decoding when unused.

        This replaces the default "PIL" transformation which decodes every file
        in the sample (including images even when return_image=False).
        """
        import io
        import json as _json

        from PIL import Image as _Image

        decoded = {}
        for key, stream in sample.items():
            if key.startswith("__"):
                decoded[key] = stream
                continue

            # Skip files we don't need
            if key not in self._needed_extensions:
                # Still check for image/caption presence detection
                if key in {".jpg", ".jpeg", ".png", ".webp", ".txt"}:
                    decoded[key] = True  # sentinel for presence check
                continue

            ext = key.rsplit(".", 1)[-1] if "." in key else ""
            if ext in ("jpg", "jpeg", "png", "webp"):
                decoded[key] = _Image.open(stream)
            elif ext == "json":
                data = stream.read() if hasattr(stream, "read") else stream
                decoded[key] = _json.loads(data)
            elif ext == "npy":
                decoded[key] = np.load(stream)
            elif ext in ("txt", "text"):
                data = stream.read() if hasattr(stream, "read") else stream
                decoded[key] = data.decode("utf-8") if isinstance(data, bytes) else data
            else:
                decoded[key] = stream

        return decoded

    def _cache_filename(self, root_dir):
        return os.path.join(
            root_dir,
            f"total_size_{self.min_image_size}_{str(self.clip_filter_threshold).replace('.', '_')}_{str(self.aesthetic_threshold).replace('.', '_')}.json",
        )

    def _estimate_filtered_size(self, roots, all_shards):
        """Estimate the effective dataset size after filtering.

        Sums cached per-root sizes. If any root is missing a cache,
        estimates by sampling from the combined dataset.
        """
        has_filters = (
            self.min_image_size > 0
            or self.clip_filter_threshold > 0
            or self.aesthetic_threshold > 0
        )

        if not has_filters:
            return sum(s["nsamples"] for s in all_shards)

        rank = _get_rank()
        is_distributed = torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1

        # Try to load cached size for each root and sum them
        total_cached = 0
        all_cached = True
        for root_dir in roots:
            cache_file = self._cache_filename(root_dir)
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "r") as f:
                        cached_size = int(json.load(f)["total_size"])
                    if cached_size >= 0:
                        print0(f"Loaded cached size for {root_dir}: {cached_size}")
                        total_cached += cached_size
                        continue
                except Exception:
                    pass
            all_cached = False

        if all_cached:
            print0(f"Total cached dataset size: {total_cached}")
            return total_cached

        # Estimate by sampling from the combined dataset
        total_raw = len(self.dataset)
        sample_size = min(10000, total_raw)
        rng = random.Random(42)
        indices = rng.sample(range(total_raw), sample_size)

        pass_count = 0
        for idx in indices:
            try:
                sample = self.dataset[idx]
                metadata = sample.get(".json", {})
                if self._passes_filter(metadata):
                    pass_count += 1
            except Exception:
                continue

        pass_rate = pass_count / max(1, len(indices))
        estimated = int(total_raw * pass_rate)
        print0(f"Estimated filtered dataset size: {estimated} (pass rate: {pass_rate:.3f})")

        # Cache per-root estimates (proportional to each root's shard count)
        if rank == 0 or not is_distributed:
            for root_dir in roots:
                cache_file = self._cache_filename(root_dir)
                if not os.path.exists(cache_file):
                    # Estimate this root's share proportionally
                    root_raw = sum(
                        s["nsamples"] for s in all_shards if s["url"].startswith(root_dir)
                    )
                    root_estimated = int(estimated * root_raw / max(1, total_raw))
                    try:
                        with open(cache_file, "w") as f:
                            json.dump({"total_size": root_estimated}, f)
                        print0(f"Saved estimated size for {root_dir}: {root_estimated}")
                    except Exception as e:
                        print0(f"Error saving size cache for {root_dir}: {e}")

        return estimated

    def _passes_filter(self, metadata):
        """Check if a sample passes the image size / quality filters."""
        if not metadata:
            return True

        width = metadata.get("width")
        height = metadata.get("height")
        if width is None or height is None:
            return True

        clip_score = metadata.get("clip_score", 0)
        aesthetic = metadata.get("aesthetic_score", 0)

        return (
            width >= self.min_image_size
            and height >= self.min_image_size
            and clip_score >= self.clip_filter_threshold
            and aesthetic >= self.aesthetic_threshold
        )

    def _get_image(self, sample):
        """Extract PIL image from sample, checking common extensions."""
        for key in [".jpg", ".jpeg", ".png", ".webp"]:
            if key in sample:
                img = sample[key]
                if hasattr(img, "convert"):
                    return img.convert("RGB")
                return img
        return None

    def _apply_synthetic_selection(self, sample, metadata):
        """Randomly swap regular embeddings/scores with synthetic ones."""
        if self.synthetic_embedding_prob <= 0 or random.random() >= self.synthetic_embedding_prob:
            return

        # Swap text embeddings
        if self.text_embedding_name:
            synthetic_key = f".synthetic_{self.text_embedding_name}_embeddings.npy"
            regular_key = f".{self.text_embedding_name}_embeddings.npy"
            if synthetic_key in sample and regular_key in sample:
                sample[regular_key] = sample[synthetic_key]

        # Swap coherence scores in metadata
        if self.coherence_scores and metadata:
            for score_name in self.coherence_scores:
                synthetic_score_key = f"synthetic_{score_name}_score"
                regular_score_key = f"{score_name}_score"
                if synthetic_score_key in metadata and regular_score_key in metadata:
                    metadata[regular_score_key] = metadata[synthetic_score_key]

    def _nearby_idx(self, idx):
        """Pick a nearby replacement index (within the same shard) to avoid LRU cache misses."""
        # Find which shard this index belongs to
        import numpy as _np
        shard_idx = _np.searchsorted(self.dataset.cum_lengths[1:], idx, side="right")
        lo = int(self.dataset.cum_lengths[shard_idx])
        hi = int(self.dataset.cum_lengths[shard_idx + 1])
        return random.randint(lo, hi - 1)

    def __getitem__(self, idx):
        max_retries = 10
        for attempt in range(max_retries):
            try:
                actual_idx = idx if attempt == 0 else self._nearby_idx(idx)
                sample = self.dataset[actual_idx]

                # Get metadata and check filters
                metadata = sample.get(".json", {})
                if not self._passes_filter(metadata):
                    continue

                # Check image/caption presence
                has_image = any(k in sample for k in [".jpg", ".jpeg", ".png", ".webp"])
                has_caption = ".txt" in sample
                if not (has_image and has_caption):
                    continue

                # Apply synthetic selection
                self._apply_synthetic_selection(sample, metadata)

                # Build output dict
                result = {}

                if self.return_image:
                    pil_image = self._get_image(sample)
                    if pil_image is None:
                        continue
                    if self.image_transforms is not None:
                        result["image"] = self.image_transforms(pil_image)
                    else:
                        result["image"] = pil_image

                if self.return_text:
                    result["text"] = sample.get(".txt", "")

                if self.vae_embedding_name_mean is not None:
                    key = f".{self.vae_embedding_name_mean}.npy"
                    if key not in sample:
                        continue
                    result[self.vae_embedding_name_mean] = torch.from_numpy(sample[key])

                if self.vae_embedding_name_std is not None:
                    key = f".{self.vae_embedding_name_std}.npy"
                    if key not in sample:
                        continue
                    result[self.vae_embedding_name_std] = torch.from_numpy(sample[key])

                if self.text_embedding_name is not None:
                    key = f".{self.text_embedding_name}_embeddings.npy"
                    if key not in sample:
                        continue
                    result[self.text_embedding_name] = torch.from_numpy(sample[key])

                if self.repa_embedding_name is not None:
                    key = f".{self.repa_embedding_name}_embeddings.npy"
                    if key not in sample:
                        continue
                    result[self.repa_embedding_name] = torch.from_numpy(sample[key])

                if self.coherence_scores:
                    for score_name in self.coherence_scores:
                        if score_name in metadata:
                            value = metadata[score_name]
                            if self.bin_coherence and score_name in self.bins:
                                bucket_index = torch.bucketize(
                                    torch.tensor(value, dtype=torch.float32),
                                    self.bins[score_name],
                                ).item()
                                value = max(0, bucket_index - 1) / (self.actual_num_bins - 1)
                        else:
                            value = float("nan")
                        result[f"{score_name}_coherence"] = torch.tensor(value, dtype=torch.float32)

                return result

            except Exception as e:
                logging.warning(f"Error processing sample {idx} (attempt {attempt}): {e}")
                continue

        # All retries exhausted -- return a minimal valid sample
        logging.error(f"All {max_retries} retries exhausted for index {idx}. Returning placeholder.")
        return self._placeholder_sample()

    def _placeholder_sample(self):
        """Return a minimal placeholder sample for when all retries fail."""
        result = {}
        if self.return_image and self.image_transforms is not None:
            from PIL import Image
            result["image"] = self.image_transforms(Image.new("RGB", (256, 256)))
        if self.return_text:
            result["text"] = ""
        if self.vae_embedding_name_mean is not None:
            result[self.vae_embedding_name_mean] = torch.zeros(4, 32, 32)
        if self.vae_embedding_name_std is not None:
            result[self.vae_embedding_name_std] = torch.zeros(4, 32, 32)
        if self.text_embedding_name is not None:
            result[self.text_embedding_name] = torch.zeros(77, 2048)
        if self.repa_embedding_name is not None:
            result[self.repa_embedding_name] = torch.zeros(77, 2048)
        if self.coherence_scores:
            for score_name in self.coherence_scores:
                result[f"{score_name}_coherence"] = torch.tensor(float("nan"), dtype=torch.float32)
        return result

    def __len__(self):
        return self.num_samples
