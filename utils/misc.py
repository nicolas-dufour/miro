import os
from typing import Optional


def dummy_value_loader(value):
    return value


def _get_rank() -> Optional[int]:
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    # None to differentiate whether an environment variable was set at all
    return None


def print0(*args, **kwargs):
    """Modified print that only prints from the master process."""
    rank = _get_rank()
    if rank is None:
        print(*args, **kwargs)
    elif rank == 0:
        print(*args, **kwargs)
    else:
        pass
