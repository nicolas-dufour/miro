import json
import logging
import os
import random
from collections import OrderedDict
from functools import partial
from multiprocessing import Value
from pathlib import Path

import braceexpand
import numpy as np
import pandas as pd
import torch
import webdataset as wds
from lightning_fabric.utilities.rank_zero import _get_rank
from PIL import Image
from torch.utils.data import Dataset, get_worker_info
from tqdm import tqdm
from webdataset.tariterators import (
    base_plus_ext,
    tar_file_expander,
    url_opener,
    valid_sample,
)
from miro.utils.misc import print0


def select_synthetic_data(sample, text_embedding_name, prob, coherence_scores=None):
    """
    Randomly select between regular and synthetic text embeddings and coherence scores.

    Args:
        sample: WebDataset sample dictionary
        text_embedding_name: Name of the text embedding to process
        prob: Probability of selecting synthetic data
        coherence_scores: List of coherence score names to process

    Returns:
        Modified sample with selected embeddings and scores
    """
    # Use synthetic data based on probability
    use_synthetic = random.random() < prob

    if use_synthetic:
        # Handle text embeddings
        synthetic_key = f"synthetic_{text_embedding_name}_embeddings.npy"
        regular_key = f"{text_embedding_name}_embeddings.npy"

        if synthetic_key in sample and regular_key in sample:
            # Replace regular embedding with synthetic
            sample[regular_key] = sample[synthetic_key]

        # Handle coherence scores in JSON
        if "json" in sample:
            try:
                json_data = sample["json"]
                type_json_data = type(json_data)
                if type_json_data == bytes:
                    json_data = json_data.decode("utf-8")

                # Parse the JSON string into a dictionary
                if isinstance(json_data, str):
                    json_data = json.loads(json_data)

                # Process each coherence score
                if coherence_scores:
                    for score_name in coherence_scores:
                        # Create score key names based on patterns in preprocess_data.py
                        synthetic_score_key = f"synthetic_{score_name}_score"
                        regular_score_key = f"{score_name}_score"

                        # If synthetic score exists, use it
                        if (
                            synthetic_score_key in json_data
                            and regular_score_key in json_data
                        ):
                            json_data[regular_score_key] = json_data[
                                synthetic_score_key
                            ]
                if type_json_data == bytes:
                    sample["json"] = json.dumps(json_data).encode("utf-8")
                else:
                    # If original was not bytes, keep it as the parsed dict
                    sample["json"] = json_data

            except (json.JSONDecodeError, Exception) as e:
                logging.warning(f"Error processing JSON for synthetic selection: {e}")

    return sample


class TextDataset(Dataset):
    def __init__(
        self,
        root,
        image_transforms=None,
        text_embedding_name=None,
        vae_embedding_name_mean=None,
        vae_embedding_name_std=None,
        return_image=True,
        return_text=True,
        coherence_scores: list[str] | None = None,
    ):
        self.root = Path(root)
        self.image_transforms = image_transforms
        self.metadata = pd.read_csv(
            root / Path("global_metadata.csv"), converters={"key": str}
        )
        self.text_embedding_name = text_embedding_name
        self.vae_embedding_name_mean = vae_embedding_name_mean
        self.vae_embedding_name_std = vae_embedding_name_std
        self.return_image = return_image
        self.return_text = return_text
        self.coherence_scores = coherence_scores

    def __getitem__(self, idx):
        return_dict = {}
        metadata = self.metadata.iloc[idx]
        key = metadata["key"]
        if self.return_image:
            image_path = self.root / "images" / f"{key}.jpg"
            image = Image.open(image_path).convert("RGB")
            if self.image_transforms is not None:
                image = self.image_transforms(image)
            return_dict["image"] = image
        if self.return_text:
            caption = metadata["caption"]
            return_dict["text"] = caption
        if self.text_embedding_name is not None:
            text_embedding_path = (
                self.root / f"{self.text_embedding_name}_embeddings" / f"{key}.npy"
            )
            text_embedding = torch.from_numpy(np.load(text_embedding_path))
            return_dict[f"{self.text_embedding_name}"] = text_embedding
        if self.vae_embedding_name_mean is not None:
            vae_embedding_path = self.root / self.vae_embedding_name_mean / f"{key}.npy"
            vae_embedding = torch.from_numpy(np.load(vae_embedding_path))
            return_dict[self.vae_embedding_name_mean] = vae_embedding
            if self.vae_embedding_name_std is not None:
                vae_embedding_std_path = (
                    self.root / self.vae_embedding_name_std / f"{key}.npy"
                )
                vae_embedding_std = torch.from_numpy(np.load(vae_embedding_std_path))
                return_dict[self.vae_embedding_name_std] = vae_embedding_std
        if self.coherence_scores:
            for score_name in self.coherence_scores:
                # Set all coherence scores to 1
                final_key = f"{score_name}_coherence"
                return_dict[final_key] = 1.0
        return return_dict

    def __len__(self):
        return len(self.metadata)


class TextWebDataset(wds.DataPipeline):
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
        coherence_scores: list[str] | None = None,
        max_samples_bin_estimation=100000,
        synthetic_embedding_prob: float = 0.0,
    ):
        self.image_transforms = image_transforms
        dataset_tar_files = []
        self.clip_filter_threshold = clip_filter_threshold
        self.coherence_scores = coherence_scores
        self.bin_coherence = bin_coherence
        self.num_bins = num_bins
        self.bin_strategy = bin_strategy
        self.refine_last_n_bins = refine_last_n_bins
        self.refine_factor = refine_factor
        self.repa_embedding_name = repa_embedding_name
        self.synthetic_embedding_prob = synthetic_embedding_prob
        self.text_embedding_name = text_embedding_name

        # Calculate actual number of bins after refinement
        if self.bin_strategy == "refined_quantile" and self.refine_last_n_bins > 0:
            self.actual_num_bins = (self.num_bins - self.refine_last_n_bins) + (
                self.refine_last_n_bins * self.refine_factor
            )
        else:
            self.actual_num_bins = self.num_bins
        # Get a list of all tar files in the directory
        if " " in root:
            root = root.split(" ")
            print0(f"Using multiple dataset[s: {root}")
        if isinstance(root, str):
            tar_files = [f for f in os.listdir(root) if f.endswith(".tar")]
            tar_files.sort()
            first_tar_file = tar_files[0].split(".")[0]
            last_tar_file = tar_files[-1].split(".")[0]
            for tar_file in tar_files:
                dataset_tar_files.append(f"{root}/{tar_file}")
            dataset_pattern = f"{root}/{{{first_tar_file}..{last_tar_file}}}.tar"
            self.num_samples, _ = get_dataset_size(
                dataset_pattern,
                min_image_size,
                clip_filter_threshold,
                aesthetic_threshold,
            )
        elif isinstance(root, list):
            num_samples = 0
            for r in root:
                tar_files = [f for f in os.listdir(r) if f.endswith(".tar")]
                tar_files.sort()
                first_tar_file = tar_files[0].split(".")[0]
                last_tar_file = tar_files[-1].split(".")[0]
                for tar_file in tar_files:
                    dataset_tar_files.append(f"{r}/{tar_file}")
                dataset_pattern_part = f"{r}/{{{first_tar_file}..{last_tar_file}}}.tar"
                num_samples_part, _ = get_dataset_size(
                    dataset_pattern_part,
                    min_image_size,
                    clip_filter_threshold,
                    aesthetic_threshold,
                )
                num_samples += num_samples_part
            self.num_samples = num_samples
        else:
            raise ValueError(
                f"root must be a string or list of strings. Got {type(root)}"
            )
        # Shuffle the dataset tar files to ensure random access across different runs
        random.shuffle(dataset_tar_files)

        rank = _get_rank()
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        is_distributed = world_size > 1

        self.bins = {}  # Initialize on all ranks
        if self.coherence_scores is not None and self.bin_coherence:
            # Determine checkpoint path
            if isinstance(root, str):
                root_path_for_checkpoint = Path(root)
            elif isinstance(root, list):
                root_path_for_checkpoint = Path(root[0])  # Use the first root
            else:
                raise ValueError("Invalid root type for checkpoint path determination")

            sorted_scores = sorted(self.coherence_scores)
            scores_str = "-".join(sorted_scores)
            # For backward compatibility: keep original filename for quantile strategy
            if self.bin_strategy == "quantile":
                checkpoint_filename = f"coherence_bins_{self.num_bins}_bins_{scores_str}_synthetic_prob_{self.synthetic_embedding_prob}.json"
            elif self.bin_strategy == "refined_quantile":
                checkpoint_filename = f"coherence_bins_{self.num_bins}_bins_{scores_str}_strategy_{self.bin_strategy}_refine_{self.refine_last_n_bins}_factor_{self.refine_factor}_synthetic_prob_{self.synthetic_embedding_prob}.json"
            else:
                checkpoint_filename = f"coherence_bins_{self.num_bins}_bins_{scores_str}_strategy_{self.bin_strategy}_synthetic_prob_{self.synthetic_embedding_prob}.json"
            coherence_bins_checkpoint_path = (
                root_path_for_checkpoint / checkpoint_filename
            )

            bins_calculated = False
            # --- Distributed Bin Calculation Logic ---
            if is_distributed:
                cache_info = {"found": False, "bins_data": None}

                # 1. Rank 0 checks cache
                if rank == 0:
                    if coherence_bins_checkpoint_path.exists():
                        print0(
                            f"Rank 0: Loading coherence bins from {coherence_bins_checkpoint_path}"
                        )
                        try:
                            with open(coherence_bins_checkpoint_path, "r") as f:
                                bins_data_loaded = json.load(f)
                            # Basic validation: Check if keys match expected scores
                            if set(bins_data_loaded.keys()) == set(
                                self.coherence_scores
                            ):
                                cache_info["found"] = True
                                cache_info["bins_data"] = bins_data_loaded
                                print0(
                                    f"Rank 0: Loaded bins from cache: {bins_data_loaded}"
                                )
                            else:
                                print0(
                                    "Rank 0: Cached bins keys mismatch. Recalculating."
                                )
                        except (json.JSONDecodeError, Exception) as e:
                            print0(
                                f"Rank 0: Error loading checkpoint {coherence_bins_checkpoint_path}: {e}. Recomputing."
                            )
                    else:
                        print0(
                            f"Rank 0: Bin cache {coherence_bins_checkpoint_path} not found. Calculating."
                        )

                # 2. Broadcast cache info from Rank 0
                broadcast_list = [cache_info]
                torch.distributed.broadcast_object_list(broadcast_list, src=0)
                received_cache_info = broadcast_list[0]

                # 3. Update self.bins if cache was found and valid
                if received_cache_info["found"]:
                    self.bins = {
                        k: torch.tensor(v, dtype=torch.float32)
                        for k, v in received_cache_info["bins_data"].items()
                    }
                    if rank != 0:
                        print0(f"Rank {rank}: Received bins from cache: {self.bins}")
                    bins_calculated = True  # Flag that bins are ready
                else:
                    # 4. Distributed Value Gathering (If No Cache)
                    if rank == 0:
                        print0("Rank 0: Starting distributed bin calculation...")
                    local_coherence_values = {
                        score: [] for score in self.coherence_scores
                    }
                    samples_per_rank = max(1, max_samples_bin_estimation // world_size)

                    # Create pipeline for gathering values, including distributed splits
                    pipeline_gather = [
                        wds.SimpleShardList(dataset_tar_files),
                        wds.split_by_node,
                        wds.split_by_worker,
                        tarfile_to_samples_nothrow,
                        wds.decode(handler=log_and_continue),
                        wds.to_tuple("json"),  # Assuming coherence scores are in json
                    ]
                    dataset_for_coherence = wds.DataPipeline(*pipeline_gather)

                    print0(
                        f"Rank {rank}: Gathering coherence values (max {samples_per_rank} samples)..."
                    )
                    num_processed = 0
                    iterable = tqdm(
                        dataset_for_coherence,
                        total=samples_per_rank,
                        desc=f"Gathering (Rank {rank})",
                        disable=rank != 0,
                    )
                    for (metadata,) in iterable:
                        if num_processed >= samples_per_rank:
                            break
                        try:
                            for score_name in self.coherence_scores:
                                if score_name in metadata:
                                    local_coherence_values[score_name].append(
                                        metadata[score_name]
                                    )
                            num_processed += 1
                        except (json.JSONDecodeError, TypeError) as e:
                            logging.warning(
                                f"Rank {rank}: Error processing metadata for binning: {e}"
                            )
                            continue  # Skip malformed samples
                    iterable.close()

                    # 5. Gather local values from all ranks to rank 0
                    gathered_values_list = [None] * world_size
                    # print0(f"Rank {rank}: Gathering objects... local values: { {k: len(v) for k,v in local_coherence_values.items()} }") # Debug
                    try:
                        torch.distributed.gather_object(
                            local_coherence_values,
                            gathered_values_list if rank == 0 else None,
                            dst=0,
                        )
                    except Exception as e:
                        print0(f"Rank {rank}: Error during gather_object: {e}")
                        # Potentially add barrier and exit or raise
                        torch.distributed.barrier()
                        raise e

                    # 6. Compute Bins, Save Cache (Rank 0 only)
                    computed_bins_data = None
                    if rank == 0:
                        print0(
                            "Rank 0: Combining gathered values and computing bins..."
                        )
                        combined_values = {score: [] for score in self.coherence_scores}
                        for rank_values in gathered_values_list:
                            if rank_values:  # Check if object was received
                                for score_name, values in rank_values.items():
                                    combined_values[score_name].extend(values)

                        temp_bins = {}
                        for score_name, values in combined_values.items():
                            if not values:
                                print0(
                                    f"Rank 0 Warning: No values found for score '{score_name}' during bin calculation. Skipping bin generation for this score."
                                )
                                continue
                            tensor_values = torch.tensor(values, dtype=torch.float32)
                            # Remove duplicates for quantile calculation stability
                            unique_values = torch.unique(tensor_values)

                            if self.bin_strategy == "uniform":
                                # Use uniform bins between min and max values
                                bins = torch.linspace(
                                    unique_values.min().item(),
                                    unique_values.max().item(),
                                    self.num_bins + 1,
                                )
                            elif self.bin_strategy == "quantile":
                                # Use quantile-based bins (uniformly filled bins)
                                if len(unique_values) < self.num_bins + 1:
                                    print0(
                                        f"Rank 0 Warning: Not enough unique values ({len(unique_values)}) for score '{score_name}' to create {self.num_bins} bins. Falling back to uniform bins."
                                    )
                                    # Fallback: Use linear space if insufficient points
                                    bins = torch.linspace(
                                        unique_values.min().item(),
                                        unique_values.max().item(),
                                        self.num_bins + 1,
                                    )
                                else:
                                    try:
                                        bins = torch.quantile(
                                            unique_values,  # Use unique values
                                            q=torch.linspace(0, 1, self.num_bins + 1),
                                        )
                                    except RuntimeError as e:
                                        print0(
                                            f"Rank 0 Warning: torch.quantile failed for '{score_name}': {e}. Falling back to uniform bins."
                                        )
                                        bins = torch.linspace(
                                            unique_values.min().item(),
                                            unique_values.max().item(),
                                            self.num_bins + 1,
                                        )
                            elif self.bin_strategy == "refined_quantile":
                                # Use refined quantile-based bins: subdivide last N bins into k sub-bins each
                                # Total bins: (num_bins - refine_last_n_bins) + (refine_last_n_bins * refine_factor)
                                total_bins_needed = (
                                    self.actual_num_bins + 1
                                )  # Number of bin edges

                                if len(unique_values) < total_bins_needed:
                                    print0(
                                        f"Rank 0 Warning: Not enough unique values ({len(unique_values)}) for score '{score_name}' to create {self.actual_num_bins} refined bins. Falling back to uniform bins."
                                    )
                                    bins = torch.linspace(
                                        unique_values.min().item(),
                                        unique_values.max().item(),
                                        total_bins_needed,
                                    )
                                else:
                                    try:
                                        # Build refined quantile positions
                                        quantile_positions = []

                                        # First part: regular bins (0 to num_bins - refine_last_n_bins)
                                        for i in range(
                                            self.num_bins - self.refine_last_n_bins + 1
                                        ):
                                            quantile_positions.append(i / self.num_bins)

                                        # Second part: refined bins for the last N bins
                                        if self.refine_last_n_bins > 0:
                                            start_q = (
                                                self.num_bins - self.refine_last_n_bins
                                            ) / self.num_bins
                                            end_q = 1.0
                                            range_q = end_q - start_q
                                            num_refined_bins = (
                                                self.refine_last_n_bins
                                                * self.refine_factor
                                            )

                                            for i in range(1, num_refined_bins + 1):
                                                q = (
                                                    start_q
                                                    + i * range_q / num_refined_bins
                                                )
                                                quantile_positions.append(q)

                                        quantile_tensor = torch.tensor(
                                            quantile_positions, dtype=torch.float32
                                        )
                                        bins = torch.quantile(
                                            unique_values, q=quantile_tensor
                                        )

                                        print0(
                                            f"Rank 0: Created {self.actual_num_bins} refined bins (last {self.refine_last_n_bins} bins subdivided by {self.refine_factor}x) for '{score_name}'"
                                        )
                                    except RuntimeError as e:
                                        print0(
                                            f"Rank 0 Warning: torch.quantile failed for refined quantiles '{score_name}': {e}. Falling back to uniform bins."
                                        )
                                        bins = torch.linspace(
                                            unique_values.min().item(),
                                            unique_values.max().item(),
                                            total_bins_needed,
                                        )
                            else:
                                raise ValueError(
                                    f"Invalid bin_strategy: {self.bin_strategy}. Must be 'quantile', 'uniform', or 'refined_quantile'."
                                )

                            # Ensure boundaries are exactly 0 and 1 if data allows
                            # Check if min/max are already close to 0/1 before forcing
                            # if tensor_values.min() <= 0.0: bins[0] = 0.0
                            # if tensor_values.max() >= 1.0: bins[-1] = 1.0
                            bins[0] = 0.0  # Force 0 lower bound
                            bins[-1] = 1.0  # Force 1 upper bound

                            # Handle potential duplicate bin edges from quantile calculation
                            bins = torch.unique(bins)
                            if len(bins) < 2:
                                print0(
                                    f"Rank 0 Error: Could not generate valid bins for '{score_name}' (only {len(bins)} unique edges). Check data distribution."
                                )
                                continue  # Skip this score if bins invalid
                            elif len(bins) <= self.num_bins:
                                print0(
                                    f"Rank 0 Warning: Generated fewer than requested bins ({len(bins)-1} bins) for '{score_name}' due to data distribution."
                                )

                            temp_bins[score_name] = bins
                            print0(
                                f"Rank 0: Computed bins for {score_name}: {temp_bins[score_name]}"
                            )

                        if temp_bins:
                            self.bins = temp_bins  # Update Rank 0's bins
                            computed_bins_data = {
                                k: v.tolist() for k, v in self.bins.items()
                            }
                            print0(
                                f"Rank 0: Saving coherence bins to {coherence_bins_checkpoint_path}"
                            )
                            try:
                                with open(coherence_bins_checkpoint_path, "w") as f:
                                    json.dump(computed_bins_data, f)
                            except Exception as e:
                                print0(
                                    f"Rank 0: Error saving checkpoint {coherence_bins_checkpoint_path}: {e}"
                                )
                        else:
                            print0("Rank 0: No bins were computed successfully.")

                    # 7. Broadcast computed bins (or signal failure)
                    broadcast_computed_bins = [
                        computed_bins_data
                    ]  # List containing dict or None
                    torch.distributed.broadcast_object_list(
                        broadcast_computed_bins, src=0
                    )

                    # 8. Receive Bins on other ranks
                    if rank != 0:
                        received_bins_data = broadcast_computed_bins[0]
                        if received_bins_data is not None:
                            self.bins = {
                                k: torch.tensor(v, dtype=torch.float32)
                                for k, v in received_bins_data.items()
                            }
                            print0(f"Rank {rank}: Received computed bins: {self.bins}")
                        else:
                            print0(
                                f"Rank {rank}: Received signal that bin computation failed on rank 0."
                            )
                            # Handle failure case if necessary, e.g., raise error or disable binning
                            self.bin_coherence = False  # Disable binning if it failed

                    bins_calculated = True  # Mark as done (or failed)

            # --- Non-Distributed Bin Calculation Logic ---
            else:  # Single process case
                if coherence_bins_checkpoint_path.exists():
                    print0(
                        f"Loading coherence bins from {coherence_bins_checkpoint_path}"
                    )
                    try:
                        with open(coherence_bins_checkpoint_path, "r") as f:
                            bins_data = json.load(f)
                        if set(bins_data.keys()) == set(self.coherence_scores):
                            self.bins = {
                                k: torch.tensor(v, dtype=torch.float32)
                                for k, v in bins_data.items()
                            }
                            print0(f"Loaded bins: {self.bins}")
                            bins_calculated = True
                        else:
                            print0("Cached bins keys mismatch. Recalculating.")
                    except (json.JSONDecodeError, FileNotFoundError, Exception) as e:
                        print0(
                            f"Error loading checkpoint {coherence_bins_checkpoint_path}: {e}. Recomputing."
                        )
                        self.bins = {}  # Reset bins

                if (
                    not bins_calculated
                ):  # Checkpoint didn't exist or failed to load/validate
                    print0("Calculating coherence bins (single process)...")
                    # Single process bin calculation logic
                    pipeline_gather = [
                        wds.SimpleShardList(dataset_tar_files),
                        wds.shuffle(1000),  # Shuffle shards for better sampling
                        tarfile_to_samples_nothrow,
                        wds.decode(handler=log_and_continue),
                        wds.map(
                            partial(
                                select_synthetic_data,
                                text_embedding_name=text_embedding_name,
                                prob=synthetic_embedding_prob,
                                coherence_scores=coherence_scores,
                            )
                        ),
                        wds.to_tuple("json"),
                    ]
                    dataset_for_coherence = wds.DataPipeline(*pipeline_gather)

                    coherence_values = {score: [] for score in self.coherence_scores}
                    num_processed = 0
                    iterable = tqdm(
                        dataset_for_coherence,
                        total=max_samples_bin_estimation,
                        desc="Gathering values",
                    )
                    for i, (metadata,) in enumerate(iterable):
                        if i >= max_samples_bin_estimation:
                            break
                        try:
                            for score_name in self.coherence_scores:
                                if score_name in metadata:
                                    coherence_values[score_name].append(
                                        metadata[score_name]
                                    )
                            num_processed += 1
                        except (json.JSONDecodeError, TypeError) as e:
                            logging.warning(
                                f"Error processing metadata for binning: {e}"
                            )
                            continue  # Skip malformed samples
                    iterable.close()

                    for score_name, values in coherence_values.items():
                        if not values:
                            print0(
                                f"Warning: No values found for score '{score_name}' during bin calculation."
                            )
                            continue
                        tensor_values = torch.tensor(values, dtype=torch.float32)
                        unique_values = torch.unique(tensor_values)

                        if self.bin_strategy == "uniform":
                            # Use uniform bins between min and max values
                            bins = torch.linspace(
                                unique_values.min().item(),
                                unique_values.max().item(),
                                self.num_bins + 1,
                            )
                        elif self.bin_strategy == "quantile":
                            # Use quantile-based bins (uniformly filled bins)
                            if len(unique_values) < self.num_bins + 1:
                                print0(
                                    f"Warning: Not enough unique values ({len(unique_values)}) for score '{score_name}' to create {self.num_bins} bins. Falling back to uniform bins."
                                )
                                bins = torch.linspace(
                                    unique_values.min().item(),
                                    unique_values.max().item(),
                                    self.num_bins + 1,
                                )
                            else:
                                try:
                                    bins = torch.quantile(
                                        unique_values,
                                        q=torch.linspace(0, 1, self.num_bins + 1),
                                    )
                                except RuntimeError as e:
                                    print0(
                                        f"Warning: torch.quantile failed for '{score_name}': {e}. Falling back to uniform bins."
                                    )
                                    bins = torch.linspace(
                                        unique_values.min().item(),
                                        unique_values.max().item(),
                                        self.num_bins + 1,
                                    )
                        elif self.bin_strategy == "refined_quantile":
                            # Use refined quantile-based bins: subdivide last N bins into k sub-bins each
                            # Total bins: (num_bins - refine_last_n_bins) + (refine_last_n_bins * refine_factor)
                            total_bins_needed = (
                                self.actual_num_bins + 1
                            )  # Number of bin edges

                            if len(unique_values) < total_bins_needed:
                                print0(
                                    f"Warning: Not enough unique values ({len(unique_values)}) for score '{score_name}' to create {self.actual_num_bins} refined bins. Falling back to uniform bins."
                                )
                                bins = torch.linspace(
                                    unique_values.min().item(),
                                    unique_values.max().item(),
                                    total_bins_needed,
                                )
                            else:
                                try:
                                    # Build refined quantile positions
                                    quantile_positions = []

                                    # First part: regular bins (0 to num_bins - refine_last_n_bins)
                                    for i in range(
                                        self.num_bins - self.refine_last_n_bins + 1
                                    ):
                                        quantile_positions.append(i / self.num_bins)

                                    # Second part: refined bins for the last N bins
                                    if self.refine_last_n_bins > 0:
                                        start_q = (
                                            self.num_bins - self.refine_last_n_bins
                                        ) / self.num_bins
                                        end_q = 1.0
                                        range_q = end_q - start_q
                                        num_refined_bins = (
                                            self.refine_last_n_bins * self.refine_factor
                                        )

                                        for i in range(1, num_refined_bins + 1):
                                            q = start_q + i * range_q / num_refined_bins
                                            quantile_positions.append(q)

                                    quantile_tensor = torch.tensor(
                                        quantile_positions, dtype=torch.float32
                                    )
                                    bins = torch.quantile(
                                        unique_values, q=quantile_tensor
                                    )

                                    print0(
                                        f"Created {self.actual_num_bins} refined bins (last {self.refine_last_n_bins} bins subdivided by {self.refine_factor}x) for '{score_name}'"
                                    )
                                except RuntimeError as e:
                                    print0(
                                        f"Warning: torch.quantile failed for refined quantiles '{score_name}': {e}. Falling back to uniform bins."
                                    )
                                    bins = torch.linspace(
                                        unique_values.min().item(),
                                        unique_values.max().item(),
                                        total_bins_needed,
                                    )
                        else:
                            raise ValueError(
                                f"Invalid bin_strategy: {self.bin_strategy}. Must be 'quantile', 'uniform', or 'refined_quantile'."
                            )

                        # bins[0] = 0.0
                        # bins[-1] = 1.0
                        bins[0] = 0.0  # Force 0 lower bound
                        bins[-1] = 1.0  # Force 1 upper bound
                        bins = torch.unique(bins)
                        if len(bins) < 2:
                            print0(
                                f"Error: Could not generate valid bins for '{score_name}' (only {len(bins)} unique edges). Check data distribution."
                            )
                            continue
                        elif len(bins) <= self.num_bins:
                            print0(
                                f"Warning: Generated fewer than requested bins ({len(bins)-1} bins) for '{score_name}' due to data distribution."
                            )

                        self.bins[score_name] = bins
                        print0(
                            f"Computed bins for {score_name}: {self.bins[score_name]}"
                        )

                    # Save the computed bins
                    if self.bins:
                        print0(
                            f"Saving coherence bins to {coherence_bins_checkpoint_path}"
                        )
                        bins_to_save = {k: v.tolist() for k, v in self.bins.items()}
                        try:
                            with open(coherence_bins_checkpoint_path, "w") as f:
                                json.dump(bins_to_save, f)
                        except Exception as e:
                            print0(
                                f"Error saving checkpoint {coherence_bins_checkpoint_path}: {e}"
                            )
                    else:
                        print0("No bins were computed.")
                    bins_calculated = True

            # --- Synchronization Barrier --- Ensures all ranks wait until bins are loaded/calculated
            if is_distributed:
                # print0(f"Rank {rank}: Reaching bin barrier...") # Debug
                torch.distributed.barrier()
                # print0(f"Rank {rank}: Passed bin barrier.") # Debug

            # Final check if bins are actually populated
            if not self.bins and self.bin_coherence:
                logging.warning(
                    "Coherence binning enabled, but no bins were successfully loaded or computed. Disabling binning for this run."
                )
                self.bin_coherence = False  # Disable if failed

        # --- Rest of the Pipeline Setup ---
        self.shared_epoch = SharedEpoch(epoch)
        pipeline = [wds.SimpleShardList(dataset_tar_files)]

        if distributed:
            pipeline.extend(
                [
                    (
                        detshuffle2(
                            bufsize=shard_shuffle_size,
                            initial=shard_shuffle_initial,
                            seed=seed,
                            epoch=self.shared_epoch,
                        )
                        if train
                        else None
                    ),
                    wds.split_by_node,
                    wds.split_by_worker,
                    tarfile_to_samples_nothrow,
                    (
                        wds.shuffle(
                            bufsize=sample_shuffle_size,
                            initial=sample_shuffle_initial,
                        )
                        if train
                        else None
                    ),
                ]
            )
        else:
            pipeline.extend(
                [
                    (
                        wds.shuffle(
                            bufsize=shard_shuffle_size,
                            initial=sample_shuffle_initial,
                        )
                        if train
                        else None
                    ),
                    wds.split_by_worker,
                    tarfile_to_samples_nothrow,
                    (
                        wds.shuffle(
                            bufsize=sample_shuffle_size,
                            initial=sample_shuffle_initial,
                        )
                        if train
                        else None
                    ),
                ]
            )

        pipeline.extend(
            [
                wds.select(filter_no_caption_or_no_image),
                wds.select(
                    partial(
                        filter_metadata,
                        min_image_size=min_image_size,
                        min_clip_score=clip_filter_threshold,
                        min_aesthetic=aesthetic_threshold,
                    )
                ),
            ]
        )

        # Add a stage to randomly select between regular and synthetic data
        if synthetic_embedding_prob > 0:
            pipeline.append(
                wds.map(
                    partial(
                        select_synthetic_data,
                        text_embedding_name=text_embedding_name,
                        prob=synthetic_embedding_prob,
                        coherence_scores=coherence_scores,
                    )
                )
            )

        outputs_transforms = OrderedDict()
        outputs_rename = OrderedDict()
        if return_image:
            outputs_rename["image.jpg"] = "jpg;png;webp;jpeg"
            outputs_transforms["image.jpg"] = self.image_transforms
        if vae_embedding_name_mean is not None:
            outputs_rename[
                f"{vae_embedding_name_mean}.npy"
            ] = f"{vae_embedding_name_mean}.npy"
            outputs_transforms[
                f"{vae_embedding_name_mean}.npy"
            ] = lambda x: torch.from_numpy(x)
        if vae_embedding_name_std is not None:
            outputs_rename[
                f"{vae_embedding_name_std}.npy"
            ] = f"{vae_embedding_name_std}.npy"
            outputs_transforms[
                f"{vae_embedding_name_std}.npy"
            ] = lambda x: torch.from_numpy(x)
        if return_text:
            outputs_rename["text.txt"] = "txt"
            outputs_transforms["text.txt"] = lambda x: x
        if text_embedding_name is not None:
            outputs_rename[
                f"{text_embedding_name}.npy"
            ] = f"{text_embedding_name}_embeddings.npy"
            outputs_transforms[
                f"{text_embedding_name}.npy"
            ] = lambda x: torch.from_numpy(x)
        if self.repa_embedding_name is not None:
            outputs_rename[
                f"{self.repa_embedding_name}.npy"
            ] = f"{self.repa_embedding_name}_embeddings.npy"
            outputs_transforms[
                f"{self.repa_embedding_name}.npy"
            ] = lambda x: torch.from_numpy(x)
        if self.coherence_scores is not None:
            for score_name in self.coherence_scores:
                final_key_intermediate = f"{score_name}_coherence.json"
                outputs_rename[
                    final_key_intermediate
                ] = "json"  # Get the whole json first, rename to final key + .json

                def _transform_coherence(
                    x,
                    score=score_name,
                    binning=self.bin_coherence,
                    bins=self.bins.get(score_name) if self.bin_coherence else None,
                    num_bins=self.actual_num_bins,
                ):
                    value = x[score]
                    if binning:
                        if bins is None:
                            # Handle case where bins weren't computed (e.g., no samples found)
                            logging.warning(
                                f"Bins for score '{score}' not found. Returning 0.0."
                            )
                            return torch.tensor(0.0, dtype=torch.float32)
                        # Bucketize and scale to [0, 1]
                        # Subtract 1 because bucketize returns 1-based index, and we want 0-based index for scaling.
                        # Clamp ensures indices are within valid range [0, num_bins-1] after subtracting 1.
                        bucket_index = torch.bucketize(
                            torch.tensor(value, dtype=torch.float32),
                            bins,
                        ).item()
                        scaled_value = (max(0, bucket_index - 1)) / (num_bins - 1)
                        return torch.tensor(scaled_value, dtype=torch.float32)

                    else:
                        return torch.tensor(value, dtype=torch.float32)

                # Assign transform to the intermediate key
                outputs_transforms[final_key_intermediate] = partial(
                    _transform_coherence
                )

        pipeline.extend(
            [
                wds.rename(**outputs_rename),
                filter_dict_keys(*outputs_rename.keys(), handler=log_and_continue),
            ]
        )
        if return_image:
            pipeline.append(wds.decode("pilrgb", handler=log_and_continue))
        else:
            pipeline.append(wds.decode("pilrgb", handler=log_and_continue))
        pipeline.extend(
            [
                wds.map_dict(**outputs_transforms, handler=log_and_continue),
                self._build_final_rename_map(outputs_transforms),
            ]
        )

        super().__init__(*pipeline)

    def _build_final_rename_map(self, outputs_transforms):
        final_rename_map = {}
        for intermediate_key in outputs_transforms.keys():
            base_key = intermediate_key.split(".")[0]
            if base_key in (
                self.coherence_scores or []
            ):  # Check if it's a coherence score
                final_key = f"{base_key}_coherence"
            else:
                final_key = (
                    base_key  # Keep original name for image, text, embeddings etc.
                )
            final_rename_map[final_key] = intermediate_key
        return wds.rename(**final_rename_map)

    def get_clip_score_from_meta(self, metadata):
        return metadata[self.clip_name]

    def __len__(self):
        return self.num_samples


# Modified from open_clip


def get_dataset_size(
    shards, min_image_size, clip_filter_threshold, aesthetic_threshold
):
    shards_list, _ = expand_urls(shards)
    if not shards_list:
        return 0, 0

    num_shards = len(shards_list)
    dir_path = os.path.dirname(shards_list[0])
    # Simplified cache filename for total size only
    cache_filename = os.path.join(
        dir_path,
        f"total_size_{min_image_size}_{str(clip_filter_threshold).replace('.', '_')}_{str(aesthetic_threshold).replace('.', '_')}.json",
    )

    rank = _get_rank()
    world_size = (
        torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    )
    is_distributed = world_size > 1

    total_size = -1
    cache_info = {"found": False, "total_size": -1}  # Use a dict for broadcast

    # Rank 0 checks the cache
    if rank == 0 or not is_distributed:
        if os.path.exists(cache_filename):
            try:
                with open(cache_filename, "r") as f:
                    data = json.load(f)
                    cached_total_size = int(data["total_size"])
                    # Basic validation (optional, could add more checks)
                    if cached_total_size >= 0:
                        cache_info["found"] = True
                        cache_info["total_size"] = cached_total_size
                        total_size = cached_total_size  # Update rank 0's total_size
                        print0(
                            f"Rank 0: Loaded total_size {total_size} from {cache_filename}"
                        )
                    else:
                        print0(
                            f"Rank 0: Invalid cache value in {cache_filename}. Recalculating."
                        )
            except (json.JSONDecodeError, KeyError, Exception) as e:
                print0(f"Rank 0: Error loading {cache_filename}: {e}. Recalculating.")
        else:
            print0(f"Rank 0: Cache file {cache_filename} not found. Calculating size.")

    # Broadcast cache status from Rank 0 to all other ranks
    if is_distributed:
        # Use broadcast_object_list for simplicity with dict
        broadcast_list = [cache_info]  # Needs to be a list
        torch.distributed.broadcast_object_list(broadcast_list, src=0)
        received_cache_info = broadcast_list[0]
        # Update total_size on non-zero ranks if cache was found
        if rank != 0 and received_cache_info["found"]:
            total_size = received_cache_info["total_size"]
            # print0(f"Rank {rank}: Received total_size {total_size} from cache.") # Debug

    # Calculate size distributively if not found in cache
    if total_size == -1:
        if is_distributed and rank == 0:
            print0("Rank 0: Starting distributed size calculation.")
        elif not is_distributed:
            print0("Rank 0: Starting size calculation.")
        # Pipeline for distributed counting (includes node/worker splitting)
        pipeline = [
            wds.SimpleShardList(shards_list),
        ]
        # Add splitting stages only if distributed
        if is_distributed:
            pipeline.extend(
                [
                    wds.split_by_node,
                    wds.split_by_worker,
                ]
            )
        else:
            pipeline.append(wds.split_by_worker)

        pipeline.extend(
            [
                tarfile_to_samples_nothrow,
                wds.select(
                    partial(
                        filter_metadata,
                        min_image_size=min_image_size,
                        min_clip_score=clip_filter_threshold,
                        min_aesthetic=aesthetic_threshold,
                    )
                ),
                wds.to_tuple("__key__"),  # Only need the key to count
            ]
        )

        dataset = wds.DataPipeline(*pipeline)

        # Each rank counts its local samples
        local_count = 0
        # Use tqdm only on rank 0 to avoid messy output
        iterable = tqdm(
            dataset,
            desc=f"Calculating size (Rank {rank})",
            disable=(rank != 0 and not is_distributed),
        )
        for _ in iterable:
            local_count += 1

        # Prepare tensor for reduction
        count_tensor = torch.tensor(local_count, dtype=torch.long)
        if torch.cuda.is_available():
            count_tensor = count_tensor.cuda()  # Move to GPU if needed for NCCL backend

        # Sum counts across all ranks using all_reduce
        if is_distributed:
            torch.distributed.all_reduce(
                count_tensor, op=torch.distributed.ReduceOp.SUM
            )
            # Ensure rank 0 also updates its tqdm description if it was used
            if rank == 0:
                iterable.set_description("Distributed calculation complete")

        # The final total size is now in count_tensor on all ranks
        total_size = count_tensor.item()

        # Rank 0 saves the newly calculated total size to cache
        if rank == 0 or not is_distributed:
            print0(f"Rank 0: Calculated total dataset size: {total_size}")
            try:
                with open(cache_filename, "w") as f:
                    json.dump({"total_size": total_size}, f)
                print0(f"Rank 0: Saved total_size to {cache_filename}")
            except Exception as e:
                print0(f"Rank 0: Failed to save total_size to {cache_filename}: {e}")

    # Synchronization barrier before returning
    if is_distributed:
        # print0(f"Rank {rank}: Reached barrier.") # Debug
        torch.distributed.barrier()
        # print0(f"Rank {rank}: Passed barrier.") # Debug

    if total_size < 0:  # Should not happen if logic is correct
        logging.error("Error: total_size calculation resulted in a negative value.")
        total_size = 0  # Fallback to 0
    elif total_size == 0:
        logging.warning(
            "Calculated dataset size is 0. Check filters, dataset paths, and shard content."
        )

    return total_size, num_shards


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split("::")
        assert len(weights) == len(
            urllist
        ), f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


# _SHARD_SHUFFLE_SIZE = 256
# _SHARD_SHUFFLE_INITIAL = 128
# _SAMPLE_SHUFFLE_SIZE = 5000
# _SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
        self,
        bufsize=1000,
        initial=100,
        seed=0,
        epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return wds.filters._shuffle(src, self.bufsize, self.initial, rng)


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


def group_by_keys_nothrow(
    data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None
):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if (
            current_sample is None
            or prefix != current_sample["__key__"]
            or suffix in current_sample
        ):
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def filter_no_caption_or_no_image(sample):
    has_caption = "txt" in sample
    has_image = (
        "png" in sample or "jpg" in sample or "jpeg" in sample or "webp" in sample
    )
    return has_caption and has_image


def filter_metadata(sample, min_image_size, min_clip_score, min_aesthetic):
    if "json" in sample.keys():
        try:  # Add try-except for robustness if json or keys are missing
            metadata = json.loads(sample["json"])
            width = metadata.get("width")
            height = metadata.get("height")
            # Use .get with default for optional scores to avoid KeyError
            clip_score = metadata.get("clip_score", 0)  # Default to 0 if missing
            aesthetic = metadata.get("aesthetic_score", 0)  # Default to 0 if missing

            if width is None or height is None:
                logging.warning(
                    f"Missing width/height in metadata for key {sample.get('__key__')}. Skipping filter."
                )
                return True  # Or False depending on desired behavior for missing dimensions

            return (
                width >= min_image_size
                and height >= min_image_size
                and clip_score >= min_clip_score
                and aesthetic >= min_aesthetic
            )
        except json.JSONDecodeError as e:
            logging.warning(
                f"JSON decode error for key {sample.get('__key__')}: {e}. Skipping filter."
            )
            return False  # Skip samples with bad JSON
        except Exception as e:  # Catch other potential errors during metadata access
            logging.warning(
                f"Error processing metadata for key {sample.get('__key__')}: {e}. Skipping filter."
            )
            return False
    else:
        return True  # Keep sample if no json is present (maybe intended?)


def _filter_dict_keys(
    data,
    *args,
    handler=wds.reraise_exception,
    missing_is_error=True,
    none_is_error=None,
):
    """Convert dict samples to tuples."""
    if none_is_error is None:
        none_is_error = missing_is_error
    if len(args) == 1 and isinstance(args[0], str) and " " in args[0]:
        args = args[0].split()

    for sample in data:
        try:
            result = {
                f: wds.getfirst(sample, f, missing_is_error=missing_is_error)
                for f in args
            }
            # Removed the check for None values as some transforms might legitimately return None temporarily
            # if none_is_error and any(x is None for x in result.values()):
            #     raise ValueError(f"to_tuple {args} got None for keys {list(sample.keys())} -> {result}")
            yield result
        except Exception as exn:
            if handler(exn):
                continue
            else:
                break


filter_dict_keys = wds.pipelinefilter(_filter_dict_keys)
