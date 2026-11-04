"""Data-parallel helpers: each rank rolls out its own prompts and divides its loss by the
global token count, so summed gradients equal the gradient of the global objective.
Every rank enters every collective.
"""
from __future__ import annotations

import datetime
import os

import torch
import torch.distributed as dist


def init() -> tuple[int, int, int]:
    """Returns (rank, world_size, local_rank); works without torchrun too."""
    if "RANK" not in os.environ:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        backend = "nccl"
    else:
        backend = "gloo"
        local = 0
    if not dist.is_initialized():
        # rollout phases desynchronize ranks by many minutes (different tool-call counts)
        dist.init_process_group(backend=backend, timeout=datetime.timedelta(hours=6))
    return rank, world, local


def barrier():
    if dist.is_initialized():
        dist.barrier()


def all_sum_vec(values: list[float], device) -> list[float]:
    """Sum a fixed-length vector of scalars across ranks in one collective."""
    if not dist.is_initialized():
        return [float(v) for v in values]
    t = torch.tensor([float(v) for v in values], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return [float(x) for x in t.tolist()]


def all_any(flag: bool, device) -> bool:
    """True on every rank when any rank reports the flag."""
    if not dist.is_initialized():
        return bool(flag)
    t = torch.tensor([1 if flag else 0], device=device, dtype=torch.int32)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(t.item())


def gather_objects(obj, world: int) -> list:
    if not dist.is_initialized():
        return [obj]
    out = [None] * world
    dist.all_gather_object(out, obj)
    return out


def shard_batch(batch: list, rank: int, world: int) -> list:
    """Deterministic per-rank slice of a globally identical prompt batch."""
    return batch[rank::world]


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()
