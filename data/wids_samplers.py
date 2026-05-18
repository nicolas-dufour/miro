"""Distributed samplers used by ``WIDSDataModule``.

Reuses ``wids.DistributedChunkedSampler`` (the chunk-balanced one) from the
public PyPI package and vendors only ``DistributedLocalSampler`` — a small
variant that gives each rank a contiguous local slice instead of a strided
one (favours filesystem locality on lustre/networked shards).
"""
from __future__ import annotations

import math

import torch
from torch.utils.data.distributed import DistributedSampler
from wids import DistributedChunkedSampler  # noqa: F401  (re-exported)


class DistributedLocalSampler(DistributedSampler):
    """``DistributedSampler`` that hands each rank a *contiguous* index range.

    Vendored from the legacy ``cad.data.wids.wids.DistributedLocalSampler`` so
    miro has no runtime dependency on the bundled ``cad/`` package.
    """

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        chunk_size = self.total_size // self.num_replicas
        begin = chunk_size * self.rank
        end = chunk_size * (self.rank + 1)
        indices = indices[begin:end]

        assert len(indices) == self.num_samples
        return iter(indices)


__all__ = ["DistributedChunkedSampler", "DistributedLocalSampler"]
