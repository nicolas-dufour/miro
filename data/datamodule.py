import math

import pytorch_lightning as L
import torch
import torch.distributed as dist
import webdataset as wds
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from miro.data.wids_samplers import DistributedChunkedSampler, DistributedLocalSampler
from miro.utils.misc import print0


class _DataLoaderWithExplicitLength(DataLoader):
    def __init__(self, dataset, length, **kwargs):
        super().__init__(dataset, **kwargs)
        self.length = length

    def __len__(self):
        return self.length


class WebdatasetDataModule(L.LightningDataModule):
    """
    Module to load image data
    """

    def __init__(
        self,
        train_dataset,
        val_dataset,
        full_batch_size,
        num_workers,
        collate_fn=default_collate,
        num_nodes=1,
        num_devices=1,
    ):
        super().__init__()
        num_devices = num_devices if type(num_devices) == int else len(num_devices)
        self.full_batch_size = full_batch_size
        self.batch_size = full_batch_size // (num_nodes * num_devices)
        print0(f"Each GPU will receive {self.batch_size} images")
        self.num_workers = num_workers
        self.world_size = num_nodes * num_devices
        self._train_dataset_builder = train_dataset
        self._val_dataset_builder = val_dataset
        self.collate_fn = collate_fn

    def setup(self, stage=None):
        self.train_dataset = self._train_dataset_builder()
        self.val_dataset = self._val_dataset_builder()
        self.train_dataset = self.train_dataset.compose(
            wds.batched(
                self.batch_size,
                partial=self.world_size > 1,
                collation_fn=self.collate_fn,
                # dict_collate_and_pad(["flan_t5_xl"], max_length=256),
            )
        )
        num_train_samples = self.train_dataset.num_samples
        if self.world_size > 1:
            self.num_train_batches = math.ceil(num_train_samples / self.full_batch_size)
            num_workers = max(1, self.num_workers)

            num_train_worker_batches = math.ceil(self.num_train_batches / num_workers)
            self.num_train_batches = num_train_worker_batches * num_workers
            num_train_samples = self.num_train_batches * self.full_batch_size

            self.train_dataset = self.train_dataset.with_epoch(
                num_train_worker_batches
            ).with_length(num_train_worker_batches)
        else:
            self.num_train_batches = math.ceil(num_train_samples / self.batch_size)

            self.train_dataset = self.train_dataset.with_epoch(
                self.num_train_batches
            ).with_length(self.num_train_batches)
        self.train_aug = self.train_dataset.image_transforms
        self.val_aug = self.val_dataset.image_transforms

    def train_dataloader(self):
        # self.train_dataset already yields pre-batched items via wds.batched
        # Use a DataLoader subclass declared at module scope that overrides __len__
        return _DataLoaderWithExplicitLength(
            self.train_dataset,
            self.num_train_batches,
            batch_size=None,
            shuffle=False,
            sampler=None,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )


class WIDSDataModule(L.LightningDataModule):
    """DataModule using WIDS datasets with DistributedChunkedSampler.

    Modeled on cad/data/datamodule.py:ImageDataModule. Uses standard
    DataLoader with proper samplers instead of webdataset batching hacks.
    """

    def __init__(
        self,
        train_dataset,
        val_dataset,
        full_batch_size,
        num_workers,
        collate_fn=default_collate,
        num_nodes=1,
        num_devices=1,
        sampler_chunksize=100000,
        prefetch_factor=4,
        sampler_seed=0,
        shard_exclusive_workers=False,
        shuffle_buffer_multiplier=2,
    ):
        super().__init__()
        num_devices = num_devices if type(num_devices) == int else len(num_devices)
        self.batch_size = full_batch_size // (num_nodes * num_devices)
        self.world_size = num_nodes * num_devices
        self.sampler_chunksize = sampler_chunksize
        self.prefetch_factor = prefetch_factor
        self.sampler_seed = sampler_seed
        self.shard_exclusive_workers = shard_exclusive_workers
        self.shuffle_buffer_multiplier = shuffle_buffer_multiplier
        print0(f"Each GPU will receive {self.batch_size} images")
        self.num_workers = num_workers
        self._train_dataset_builder = train_dataset
        self._val_dataset_builder = val_dataset
        self.collate_fn = collate_fn

    def setup(self, stage=None):
        self.train_dataset = self._train_dataset_builder()
        self.val_dataset = self._val_dataset_builder()
        print0(f"Train dataset size: {len(self.train_dataset)}")
        print0(f"Val dataset size: {len(self.val_dataset)}")
        self.train_aug = self.train_dataset.image_transforms
        self.val_aug = self.val_dataset.image_transforms

    def _aligned_num_samples(self, dataset_len, align_to):
        """Floor dataset length to a multiple of align_to to keep all ranks in sync."""
        if align_to <= 0:
            return dataset_len
        aligned = (dataset_len // align_to) * align_to
        return aligned if aligned > 0 else dataset_len

    def _get_dist_info(self):
        """Get actual distributed rank and world_size from the process group.

        Config-derived self.world_size can be stale when devices=auto or
        num_nodes is overridden, so we always query the live process group.
        """
        if dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return 0, 1

    def train_dataloader(self):
        rank, world_size = self._get_dist_info()
        batch_size = self.batch_size if world_size == self.world_size else (
            self.batch_size * self.world_size // world_size
        )

        # DistributedChunkedSampler uses ceil division to split samples across
        # ranks: per_rank = ceil(total / world_size). The last rank can get
        # fewer samples. With drop_last=True, this means fewer batches on the
        # last rank → DDP deadlock.
        #
        # Fix: compute per_rank as an exact multiple of batch_size, then
        # set total = per_rank * world_size so all ranks are identical.
        raw_len = len(self.train_dataset)
        per_rank = raw_len // world_size
        per_rank = (per_rank // batch_size) * batch_size  # floor to batch multiple
        train_num_samples = per_rank * world_size

        print0(
            f"Train dataloader: world_size={world_size}, batch_size={batch_size}, "
            f"per_rank={per_rank}, total={train_num_samples} (from {raw_len}), "
            f"batches_per_rank={per_rank // batch_size}"
        )

        if self.shard_exclusive_workers:
            from miro.data.text_wids_dataset import ShardExclusiveSampler

            sampler = ShardExclusiveSampler(
                self.train_dataset,
                num_workers=self.num_workers,
                batch_size=batch_size,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=self.sampler_seed,
            )
        else:
            sampler = DistributedChunkedSampler(
                self.train_dataset,
                num_replicas=world_size,
                num_samples=train_num_samples,
                rank=rank,
                shuffle=True,
                chunksize=self.sampler_chunksize,
                seed=self.sampler_seed,
            )

        dataloader_kwargs = {}
        if self.num_workers > 0:
            dataloader_kwargs["prefetch_factor"] = self.prefetch_factor

        dl = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 1,
            drop_last=True,
            **dataloader_kwargs,
        )

        if self.shard_exclusive_workers:
            from miro.data.text_wids_dataset import ShuffledBatchLoader

            dl = ShuffledBatchLoader(
                dl, buffer_batches=self.num_workers * self.shuffle_buffer_multiplier
            )

        return dl

    def val_dataloader(self):
        rank, world_size = self._get_dist_info()
        batch_size = self.batch_size if world_size == self.world_size else (
            self.batch_size * self.world_size // world_size
        )
        sampler = DistributedLocalSampler(
            self.val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=True,
        )

        dataloader_kwargs = {}
        if self.num_workers > 0:
            dataloader_kwargs["prefetch_factor"] = self.prefetch_factor

        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 1,
            **dataloader_kwargs,
        )

    def test_dataloader(self):
        return self.val_dataloader()


def dict_collate_and_pad(keys_to_pad, max_length):
    def dict_collate(batch):
        output_dict = {}
        if isinstance(batch[0], dict):
            for key in batch[0].keys():
                list_key = [d[key] for d in batch]
                if key not in keys_to_pad:
                    output_dict[key] = dict_collate(list_key)
                else:
                    output_dict[f"{key}_embeddings"] = torch.zeros(
                        len(list_key),
                        max_length,
                        list_key[0].shape[-1],
                        dtype=list_key[0].dtype,
                    )
                    output_dict[f"{key}_mask"] = torch.zeros(
                        len(list_key), max_length, dtype=torch.bool
                    )
                    for i, x in enumerate(list_key):
                        output_dict[f"{key}_embeddings"][
                            i, : min(len(x), max_length)
                        ] = x[:max_length]
                        output_dict[f"{key}_mask"][i, : min(len(x), max_length)] = 1
        else:
            return default_collate(batch)
        return output_dict

    return dict_collate


def collate_to_dict(keys):
    key_set = set(keys)

    def collate(batch):
        if len(batch) == 0:
            return {}

        # Filter each sample first, then collate only retained keys.
        if isinstance(batch[0], dict):
            missing_keys = key_set.difference(batch[0].keys())
            if missing_keys:
                raise KeyError(
                    f"Missing keys {sorted(missing_keys)} in batch sample. Available keys: {list(batch[0].keys())}"
                )
            filtered_batch = [{k: sample[k] for k in sample if k in key_set} for sample in batch]
            return default_collate(filtered_batch)

        collated_batch = default_collate(batch)
        output_dict = {}
        for i, key in enumerate(keys):
            output_dict[key] = collated_batch[i]
        return output_dict

    return collate
