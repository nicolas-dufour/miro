# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# This file is adapted from https://github.com/NVlabs/Sana/tree/main/diffusion/data/wids
import base64
import gzip
import hashlib
import io
import json
import math
import os
import os.path as osp
import random
import sqlite3
import uuid
import warnings
from functools import lru_cache, partial
from typing import Any, BinaryIO, Dict, List, Optional, Sequence, TypeVar, Union
from urllib.parse import quote, urlparse

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from .wids_dl import download_and_open
from .wids_lru import LRUCache
from .wids_mmtar import MMIndexedTar
from .wids_specs import load_dsdesc_and_resolve, urldir
from .wids_tar import TarFileReader, find_index_file

try:
    from torch.utils.data import Dataset, Sampler
except ImportError:

    class Dataset:
        pass

    class Sampler:
        pass


T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


def compute_file_md5sum(fname: Union[str, BinaryIO], chunksize: int = 1000000) -> str:
    """Compute the md5sum of a file in chunks."""
    md5 = hashlib.md5()
    if isinstance(fname, str):
        with open(fname, "rb") as f:
            for chunk in iter(lambda: f.read(chunksize), b""):
                md5.update(chunk)
    else:
        fname.seek(0)
        for chunk in iter(lambda: fname.read(chunksize), b""):
            md5.update(chunk)
    return md5.hexdigest()


def compute_num_samples(fname):
    ds = IndexedTarSamples(fname)
    return len(ds)


def splitname(fname):
    """Returns the basename and extension of a filename.
    
    The key (basename) is everything before the first dot,
    and the extension is everything after (including the dot).
    This follows webdataset conventions for multi-extension files.
    
    Example:
        splitname("000000000.vae_embeddings_mean_256.npy")
        -> ("000000000", ".vae_embeddings_mean_256.npy")
    """
    assert "." in fname, "Filename must have an extension"
    first_dot = fname.index(".")
    basename = fname[:first_dot]
    extension = fname[first_dot:]
    return basename, extension


def group_by_key(names):
    """Group the file names by key.

    Args:
        names: A list of file names.

    Returns:
        A list of lists of indices, where each sublist contains indices of files
        with the same key.
    """
    groups = []
    kmaps = {}
    for i, fname in enumerate(names):
        # Ignore files that are not in a subdirectory.
        if "." not in fname:
            print(f"Warning: Ignoring file {fname} (no '.')")
            continue
        if fname == ".":
            print(f"Warning: Ignoring the '.' file.")
            continue
        key, ext = splitname(fname)
        if key not in kmaps:
            kmaps[key] = []
        kmaps[key].append(i)
    for k, v in kmaps.items():
        groups.append(v)
    return groups


def default_decoder(sample: Dict[str, Any], format: Optional[Union[bool, str]] = True):
    """A default decoder for webdataset.

    This handles common file extensions: .txt, .cls, .cls2,
        .jpg, .png, .json, .npy, .mp, .pt, .pth, .pickle, .pkl.
    These are the most common extensions used in webdataset.
    For other extensions, users can provide their own decoder.

    Args:
        sample: sample, modified in place
    """
    sample = dict(sample)
    for key, stream in sample.items():
        extensions = key.split(".")
        if len(extensions) < 1:
            continue
        extension = extensions[-1]
        if extension in ["gz"]:
            decompressed = gzip.decompress(stream.read())
            stream = io.BytesIO(decompressed)
            if len(extensions) < 2:
                sample[key] = stream
                continue
            extension = extensions[-2]
        if key.startswith("__"):
            continue
        elif extension in ["txt", "text"]:
            value = stream.read()
            sample[key] = value.decode("utf-8")
        elif extension in ["cls", "cls2"]:
            value = stream.read()
            sample[key] = int(value.decode("utf-8"))
        elif extension in ["jpg", "jpeg", "png", "ppm", "pgm", "pbm", "pnm"]:
            if format == "PIL":
                import PIL.Image

                sample[key] = PIL.Image.open(stream)
            elif format == "numpy":
                import numpy as np

                sample[key] = np.asarray(PIL.Image.open(stream))
            else:
                raise ValueError(f"Unknown format: {format}")
        elif extension == "json":
            import json

            value = stream.read()
            sample[key] = json.loads(value)
        elif extension == "npy":
            import numpy as np

            sample[key] = np.load(stream)
        elif extension == "mp":
            import msgpack

            value = stream.read()
            sample[key] = msgpack.unpackb(value, raw=False)
        elif extension in ["pt", "pth"]:
            import torch

            sample[key] = torch.load(stream)
        elif extension in ["pickle", "pkl"]:
            import pickle

            sample[key] = pickle.load(stream)
        elif extension == "mp4":
            sample[key] = io.BytesIO(stream.read())
    return sample


def update_dict_with_extend(original_dict, update_dict):
    for key, value in update_dict.items():
        if key in original_dict and isinstance(original_dict[key], list) and isinstance(value, list):
            original_dict[key].extend(value)
        else:
            original_dict[key] = value


open_itfs = {}


class IndexedTarSamples:
    """A class that accesses samples in a tar file. The tar file must follow
    WebDataset conventions. The tar file is indexed when the IndexedTarSamples
    object is created. The samples are accessed by index using the __getitem__
    method. The __getitem__ method returns a dictionary containing the files
    for the sample. The key for each file is the extension of the file name.
    The key "__key__" is reserved for the key of the sample (the basename of
    each file without the extension).
    """

    def __init__(
        self,
        *,
        path=None,
        stream=None,
        md5sum=None,
        expected_size=None,
        use_mmap=True,
        index_file=find_index_file,
    ):
        assert path is not None or stream is not None

        # Create TarFileReader object to read from tar_file
        self.path = path
        stream = self.stream = stream or open(path, "rb")

        # verify the MD5 sum
        if md5sum is not None:
            stream.seek(0)
            got = compute_file_md5sum(stream)
            assert got == md5sum, f"MD5 sum mismatch: expected {md5sum}, got {got}"
            stream.seek(0)

        # use either the mmap or the stream based implementation
        if use_mmap:
            self.reader = MMIndexedTar(stream)
        else:
            self.reader = TarFileReader(stream, index_file=index_file)

        # Get list of all files in stream
        all_files = self.reader.names()

        # Group files by key into samples
        self.samples = group_by_key(all_files)

        # check that the number of samples is correct
        if expected_size is not None:
            assert len(self) == expected_size, f"Expected {expected_size} samples, got {len(self)}"

        self.uuid = str(uuid.uuid4())

    def close(self):
        self.reader.close()
        if not self.stream.closed:
            self.stream.close()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Get indexes of files for the sample at index idx
        try:
            indexes = self.samples[idx]
        except IndexError as e:
            print(f"[wids-debug] curr idx: {idx}, total sample length: {len(self.samples)} {e}")
            raise e
        sample = {}
        key = None
        for i in indexes:
            # Get filename and data for the file at index i
            fname, data = self.reader.get_file(i)
            # Split filename into key and extension
            k, ext = splitname(fname)
            # Make sure all files in the sample have the same key
            key = key or k
            assert key == k
            sample[ext] = data
        sample["__key__"] = key
        return sample


def hash_localname(dldir="/tmp/_wids_cache"):
    os.makedirs(dldir, exist_ok=True)

    def f(shard):
        """Given a URL, return a local name for the shard."""
        if shard.startswith("pipe:"):
            # uuencode the entire URL string
            hex = base64.urlsafe_b64encode(shard.encode()).decode()
            return os.path.join(dldir, "pipe__" + hex)
        else:
            # we hash the host and directory components into a 16 character string
            parsed = urlparse(shard)
            host_hash = hashlib.md5(str((parsed.netloc + parsed.path)).encode()).hexdigest()[:16]
            # the cache name is the host hash plus the filename
            cachename = host_hash + "__" + os.path.basename(parsed.path)
            return os.path.join(dldir, cachename)

    return f


def lru_json_load(fname):
    """Load a JSON file, caching the result.
    
    Note: This function does not use an actual LRU cache in this implementation.
    """
    with open(fname) as f:
        return json.load(f)


class LRUShards:
    """A class that manages a cache of shards.

    The cache is an LRU cache that stores the shards as IndexedTarSamples objects.
    """

    def __init__(self, lru_size, keep=False, localname=hash_localname()):
        self.localname = localname
        self.lru = LRUCache(lru_size, release_handler=self.release_handler)
        self.keep = keep

    def release_handler(self, key, value):
        value.close()

    def clear(self):
        self.lru.clear()

    def get_shard(self, url):
        assert isinstance(url, str)
        shard = self.lru[url]
        if shard is None:
            local = self.localname(url)
            with download_and_open(url, local) as stream:
                shard = IndexedTarSamples(stream=stream)
            self.lru[url] = shard
        return shard


def interpret_transformations(transformations):
    """Interpret the transformations argument.

    This takes care of transformations specified as string shortcuts
    and returns a list of callables.
    """
    if not isinstance(transformations, list):
        transformations = [transformations]

    result = []

    for transformation in transformations:
        if transformation == "PIL":
            transformation = partial(default_decoder, format="PIL")
        elif transformation == "numpy":
            transformation = partial(default_decoder, format="numpy")
        elif callable(transformation):
            pass
        else:
            raise ValueError(f"Unknown transformation: {transformation}")
        result.append(transformation)

    return result


class ShardListDataset(Dataset):
    """An indexable dataset based on a list of shards.

    The dataset is either given as a list of shards with optional options and name,
    or as a URL pointing to a JSON descriptor file.

    Datasets come in different orders:

        - original: the order in which the data was written
        - shuffled: the order in which the samples are accessed is shuffled within each shard
        - shardshuffled: the order in which the shards are accessed is shuffled (but samples within each shard are not shuffled)

    Shuffling requires random access to the dataset.
    
    Note that shard and sample shuffling are handled by the samplers, not by the dataset itself.
    """

    def __init__(
        self,
        shards=None,
        *,
        dataset_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
        lru_size: int = 10,
        localname: Optional[callable] = None,
        transformations: Union[Any, List[Any]] = "PIL",
        keep: bool = False,
        base: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        """Create a ShardListDataset.

        Args:
            shards: a list of (url, length) pairs, a ## Sana index file, or a URL to a JSON descriptor
            dataset_name: optional name for the dataset
            cache_dir: directory for caching downloaded shards
            lru_size: number of shards to keep in the LRU cache
            localname: a function that takes a URL and returns a local filename
            transformations: a list of transformations to apply to each sample
            keep: whether to keep the downloaded shards in the cache
            base: base URL for relative URLs in the shardlist
            options: options to pass to the dataset descriptor resolver
        """
        super(ShardListDataset, self).__init__()

        if options is None:
            options = {}

        # set up the cache
        self.cache_dir = cache_dir or os.environ.get("WIDS_CACHE_DIR", "/tmp/_wids_cache")
        self.localname = localname or hash_localname(dldir=self.cache_dir)

        # load the shard list
        if isinstance(shards, str):
            if shards.endswith(".json"):
                self.spec = load_dsdesc_and_resolve(shards, options=options, base=base)
                self.shards = self.spec["shardlist"]
            else:
                # assume it's a file with one shard URL per line
                with open(shards) as f:
                    self.shards = [{"url": line.strip()} for line in f if line.strip()]
                self.spec = {"shardlist": self.shards}
        elif isinstance(shards, list):
            if len(shards) > 0 and isinstance(shards[0], dict):
                self.shards = shards
            else:
                # assume it's a list of (url, nsamples) tuples
                self.shards = [{"url": url, "nsamples": n} for url, n in shards]
            self.spec = {"shardlist": self.shards, "wids_version": 1}
        else:
            raise ValueError(f"Invalid shards type: {type(shards)}")

        # compute the lengths
        self.lengths = [shard.get("nsamples") for shard in self.shards]

        # build sample index
        self.total_length = sum(self.lengths)
        self.cum_lengths = np.cumsum([0] + self.lengths)

        # set the base
        self.base = self.spec.get("base", base)

        # build info string
        self.data_info = (
            f"base: {self.base,}, name: {self.spec.get('name')}, "
            f"nfiles: {str(len(self.shards))}"
        )
        if True or int(os.environ.get("WIDS_VERBOSE", 0)):
            nbytes = sum(shard.get("filesize", 0) for shard in self.shards)
            nsamples = sum(shard["nsamples"] for shard in self.shards)
            self.data_info += f"nbytes: {str(nbytes)}, samples: {str(nsamples),}, cache: {self.cache_dir} "
        self.transformations = interpret_transformations(transformations)

        if lru_size > 200:
            warnings.warn("LRU size is very large; consider reducing it to avoid running out of file descriptors")
        self.cache = LRUShards(lru_size, localname=self.localname, keep=keep)

    def __len__(self):
        return self.total_length

    def __getitem__(self, index):
        # Find which shard the index belongs to
        shard_idx = np.searchsorted(self.cum_lengths[1:], index, side="right")
        local_idx = index - self.cum_lengths[shard_idx]

        # Get the shard
        shard_info = self.shards[shard_idx]
        url = shard_info["url"]
        shard = self.cache.get_shard(url)

        # Get the sample
        sample = shard[local_idx]

        # Add shard information
        sample["__shard__"] = url
        sample["__shardindex__"] = shard_idx
        sample["__localindex__"] = local_idx
        sample["__globalindex__"] = index

        # Apply transformations
        for transform in self.transformations:
            sample = transform(sample)

        return sample
    
    def add_transform(self, transform):
        """Add a transformation to the dataset.
        
        Returns self for chaining.
        """
        self.transformations.append(transform)
        return self


class ShardListDatasetMulti(ShardListDataset):
    """A ShardListDataset that supports multiple dataset specs."""

    def __init__(self, shards_list, **kwargs):
        """Create a ShardListDatasetMulti from multiple shard specifications.

        Args:
            shards_list: a list of shard specifications (each as would be passed to ShardListDataset)
            **kwargs: additional arguments passed to ShardListDataset
        """
        # Combine all shards from all specs
        combined_shards = []
        for shards in shards_list:
            if isinstance(shards, str):
                if shards.endswith(".json"):
                    spec = load_dsdesc_and_resolve(shards, options=kwargs.get("options"), base=kwargs.get("base"))
                    combined_shards.extend(spec["shardlist"])
                else:
                    with open(shards) as f:
                        combined_shards.extend([{"url": line.strip()} for line in f if line.strip()])
            elif isinstance(shards, list):
                if len(shards) > 0 and isinstance(shards[0], dict):
                    combined_shards.extend(shards)
                else:
                    combined_shards.extend([{"url": url, "nsamples": n} for url, n in shards])

        super().__init__(combined_shards, **kwargs)


def split_and_recombine(lst, n):
    from collections import OrderedDict

    def extract_prefix(i):
        return i["url"].split("/")[-2]

    unique_parts = list(OrderedDict((extract_prefix(item), None) for item in lst).keys())
    split_dict = {part: [] for part in unique_parts}

    for part in unique_parts:
        part_list = [item for item in lst if extract_prefix(item) == part]
        chunk_size = max(1, len(part_list) // n)
        chunks = [part_list[i * chunk_size : (i + 1) * chunk_size] for i in range(n)]

        if len(part_list) % n != 0:
            chunks[-1].extend(part_list[n * chunk_size :])

        split_dict[part] = chunks

    recombined_list = []
    for i in range(n):
        for part in unique_parts:
            recombined_list.extend(split_dict[part][i])

    return recombined_list


def lengths_to_ranges(lengths):
    """Convert a list of lengths to a list of ranges."""
    ranges = []
    start = 0
    for length in lengths:
        ranges.append((start, start + length))
        start += length
    return ranges


def intersect_range(a, b):
    """Return the intersection of the two half-open integer intervals."""
    result = max(a[0], b[0]), min(a[1], b[1])
    if result[0] >= result[1]:
        return None
    return result


def intersect_ranges(rangelist, r):
    """Return the intersection of the half-open integer interval r with the list of half-open integer intervals."""
    result = []
    for a in rangelist:
        x = intersect_range(a, r)
        if x is not None:
            result.append(x)
    return result


def iterate_ranges(ranges, rng, indexshuffle=True, shardshuffle=True):
    """Iterate over the ranges in a random order."""
    shard_indexes = list(range(len(ranges)))
    if shardshuffle:
        rng.shuffle(shard_indexes)
    for i in shard_indexes:
        lo, hi = ranges[i]
        sample_indexes = list(range(lo, hi))
        if indexshuffle:
            rng.shuffle(sample_indexes)
        yield from sample_indexes


class ShardListSampler(Sampler):
    """A sampler that samples consistent with a ShardListDataset.

    This sampler preserves locality by shuffling at the shard level
    and then within each shard.
    """

    def __init__(self, dataset, *, lengths=None, seed=0, shufflefirst=False):
        if lengths is None:
            lengths = list(dataset.lengths)
        self.ranges = lengths_to_ranges(lengths)
        self.seed = seed
        self.shufflefirst = shufflefirst
        self.epoch = 0

    def __iter__(self):
        self.rng = random.Random(self.seed + 1289738273 * self.epoch)
        shardshuffle = self.shufflefirst or self.epoch > 0
        yield from iterate_ranges(self.ranges, self.rng, shardshuffle=shardshuffle)
        self.epoch += 1


ShardedSampler = ShardListSampler


class ChunkedSampler(DistributedSampler):
    """A sampler that samples in chunks and then shuffles the samples within each chunk.

    This preserves locality of reference while still shuffling the data.
    
    Inherits from DistributedSampler so that PyTorch Lightning recognizes it
    as a distributed sampler and doesn't replace it with its own.
    """

    def __init__(
        self,
        dataset,
        *,
        num_samples=None,
        chunksize=2000,
        seed=0,
        shuffle=False,
        shufflefirst=False,
        num_replicas=1,
        rank=0,
    ):
        # Initialize DistributedSampler minimally - we override all its behavior
        # but Lightning checks isinstance() so we need the inheritance
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = False
        
        if isinstance(num_samples, int):
            lo, hi = 0, num_samples
        elif num_samples is None:
            lo, hi = 0, len(dataset)
        else:
            lo, hi = num_samples
        self.ranges = [(i, min(i + chunksize, hi)) for i in range(lo, hi, chunksize)]
        self._num_samples = hi - lo
        self.seed = seed
        self.shuffle = shuffle
        self.shufflefirst = shufflefirst
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        self.rng = random.Random(self.seed + 1289738273 * self.epoch)
        shardshuffle = self.shufflefirst or self.epoch > 0
        yield from iterate_ranges(
            self.ranges,
            self.rng,
            indexshuffle=self.shuffle,
            shardshuffle=(self.shuffle and shardshuffle),
        )
        self.epoch += 1

    def __len__(self):
        return self._num_samples


def DistributedChunkedSampler(
    dataset: Dataset,
    *,
    num_replicas: Optional[int] = None,
    num_samples: Optional[int] = None,
    rank: Optional[int] = None,
    shuffle: bool = True,
    shufflefirst: bool = False,
    seed: int = 0,
    drop_last: bool = None,
    chunksize: int = 1000000,
) -> ChunkedSampler:
    """Return a ChunkedSampler for the current worker in distributed training.

    Reverts to a simple ChunkedSampler if not running in distributed mode.
    """
    if drop_last is not None:
        warnings.warn("DistributedChunkedSampler does not support drop_last, thus it will be ignored")
    if not dist.is_initialized():
        warnings.warn("DistributedChunkedSampler is called without distributed initialized; assuming single process")
        num_replicas = 1
        rank = 0
    else:
        num_replicas = num_replicas or dist.get_world_size()
        rank = rank or dist.get_rank()
    assert rank >= 0 and rank < num_replicas

    num_samples = num_samples or len(dataset)
    worker_chunk = (num_samples + num_replicas - 1) // num_replicas
    worker_start = rank * worker_chunk
    worker_end = min(worker_start + worker_chunk, num_samples)
    return ChunkedSampler(
        dataset,
        num_samples=(worker_start, worker_end),
        chunksize=chunksize,
        seed=seed,
        shuffle=shuffle,
        shufflefirst=shufflefirst,
        num_replicas=num_replicas,
        rank=rank,
    )


class DistributedRangedSampler(Sampler):
    """A sampler that samples in chunks and then shuffles the samples within each chunk.

    This preserves locality of reference while still shuffling the data.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        num_samples: Optional[int] = None,
        rank: Optional[int] = None,
        drop_last: bool = None,
    ):
        if drop_last is not None:
            warnings.warn("DistributedChunkedSampler does not support drop_last, thus it will be ignored")
        if not dist.is_initialized():
            warnings.warn(
                "DistributedChunkedSampler is called without distributed initialized; assuming single process"
            )
            num_replicas = 1
            rank = 0
        else:
            num_replicas = num_replicas or dist.get_world_size()
            rank = rank or dist.get_rank()
        assert rank >= 0 and rank < num_replicas
        num_samples = num_samples or len(dataset)
        self.worker_chunk = num_samples // num_replicas
        self.worker_start = rank * self.worker_chunk
        self.worker_end = min((rank + 1) * self.worker_chunk, num_samples)
        self.ranges = range(self.worker_start, self.worker_end)
        self.epoch = 0
        self.step_start = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.ranges)

    def set_start(self, start):
        self.step_start = start

    def __iter__(self):
        yield from self.ranges[self.step_start :]
        self.epoch += 1


class DistributedLocalSampler(DistributedSampler):
    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample using local chunks instead of strided
        chunk_size = self.total_size // self.num_replicas
        begin_idx = chunk_size * self.rank
        stop_idx = chunk_size * (self.rank + 1)
        indices = indices[begin_idx:stop_idx]

        assert len(indices) == self.num_samples
        return iter(indices)
