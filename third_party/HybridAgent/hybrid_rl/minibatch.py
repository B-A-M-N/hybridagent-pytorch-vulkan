"""Optimizer minibatches: every complete prompt group is assigned to exactly one of the
minibatches of a rollout batch, through a shared global slot plan so all ranks agree.
"""
from __future__ import annotations

import random
from collections.abc import Sequence

from hybrid_rl.trajectory import HybridTrajectory


def build_minibatches(
    groups: Sequence[list[HybridTrajectory]],
    num_minibatches: int,
    seed: int,
    *,
    rank: int = 0,
    world_size: int = 1,
    local_group_indices: Sequence[int] | None = None,
    groups_per_rank: int | None = None,
) -> list[list[HybridTrajectory]]:
    if num_minibatches < 1:
        raise ValueError("num_minibatches must be >= 1")
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError(f"invalid rank/world {rank}/{world_size}")

    if local_group_indices is None:
        local_group_indices = list(range(len(groups)))
    else:
        local_group_indices = [int(index) for index in local_group_indices]
    if len(local_group_indices) != len(groups) or len(set(local_group_indices)) != len(local_group_indices):
        raise ValueError("local_group_indices must align one-to-one with groups")

    inferred = max(local_group_indices, default=-1) + 1
    per_rank = int(groups_per_rank) if groups_per_rank is not None else max(inferred, len(groups))
    if per_rank < inferred:
        raise ValueError("groups_per_rank is smaller than a supplied local group index")
    global_slots = per_rank * world_size
    if global_slots and num_minibatches > global_slots:
        raise ValueError(f"num_minibatches={num_minibatches} exceeds prompt groups={global_slots}")

    # shuffle the global slots once (same seed on every rank), then round-robin
    order = list(range(global_slots))
    random.Random(int(seed)).shuffle(order)
    slot_to_minibatch = {slot: offset % num_minibatches for offset, slot in enumerate(order)}
    minibatches: list[list[HybridTrajectory]] = [[] for _ in range(num_minibatches)]
    for local_index, group in zip(local_group_indices, groups):
        minibatches[slot_to_minibatch[rank + world_size * local_index]].extend(group)
    return minibatches
