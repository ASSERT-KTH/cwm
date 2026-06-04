# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
from datetime import timedelta
from typing import Any
import os

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

COLLECTIVE_TIMEOUT = timedelta(minutes=10)


@dataclass
class DistState:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    dp_size: int
    tp_size: int
    dp_rank: int
    tp_rank: int
    dp_group: Any | None = None
    device_mesh: DeviceMesh | None = None

    @property
    def is_rank_zero(self) -> bool:
        return self.rank == 0

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def dp_enabled(self) -> bool:
        return self.dp_size > 1

    @property
    def tp_enabled(self) -> bool:
        return self.tp_size > 1


def init_distributed(*, tp_size: int, dp_size: int = 0) -> DistState:
    if tp_size <= 0:
        raise ValueError("tp_size must be positive")
    if dp_size < 0:
        raise ValueError("dp_size must be non-negative")

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        # subgroups don't inherit this timeout; override both module defaults
        # (backend string picks one) so dp/tp subgroups use it too.
        c10d.default_pg_nccl_timeout = COLLECTIVE_TIMEOUT
        c10d.default_pg_timeout = COLLECTIVE_TIMEOUT
        dist.init_process_group(backend="nccl", timeout=COLLECTIVE_TIMEOUT)
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    else:
        rank = 0
        local_rank = 0

    if world_size % tp_size != 0:
        raise ValueError(
            "WORLD_SIZE must be divisible by tp_size: "
            f"{world_size=} {tp_size=}"
        )
    inferred_dp_size = world_size // tp_size
    dp_size = dp_size or inferred_dp_size
    if dp_size != inferred_dp_size:
        raise ValueError(
            "dp_size must match WORLD_SIZE / tp_size: "
            f"{dp_size=} {world_size=} {tp_size=}"
        )
    if tp_size > 1 and world_size == 1:
        raise ValueError("tp_size > 1 requires launching with torchrun/srun tasks.")

    dp_rank = rank // tp_size
    tp_rank = rank % tp_size
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    device_mesh = None
    dp_group = None
    if world_size > 1:
        device_mesh = init_device_mesh(
            "cuda",
            (dp_size, tp_size),
            mesh_dim_names=("dp", "tp"),
        )
        if dp_size > 1:
            dp_group = device_mesh["dp"].get_group()

    return DistState(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        dp_size=dp_size,
        tp_size=tp_size,
        dp_rank=dp_rank,
        tp_rank=tp_rank,
        dp_group=dp_group,
        device_mesh=device_mesh,
    )


def destroy_distributed(state: DistState) -> None:
    if state.enabled:
        dist.destroy_process_group()


def sync_gradients(params: list[torch.nn.Parameter], state: DistState) -> None:
    if not state.dp_enabled:
        return
    for param in params:
        # all ranks must all_reduce the SAME param set; a grad that is None on
        # one rank but present on another desyncs collectives and deadlocks NCCL.
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        # SUM, not mean: each rank's loss is already scaled by local/global count
        # (see CodiModel._dp_loss_weights), so summed grads = global-mean grad.
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=state.dp_group)
