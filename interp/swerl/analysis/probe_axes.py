# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Partition TrajectoryRecords along the three analysis axes for probe training.

The three axis functions all return the same uniform shape so the probe
training loop can call them without branching:

    list of (acts: list[Tensor[dim]], labels: list[int])

where each entry in the outer list is one "bucket" on the x-axis.

Axes:
  1. bin_by_token_position  — x = normalised token position bin (0 … n_bins-1)
  2. bin_by_tool_call_index — x = normalised turn-index bucket (0 … n_buckets-1)
  3. pool_by_stage          — x = stage string key (exploration / testing / …)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class TrajectoryRecord:
    """All per-trajectory data needed for probe-axis analysis."""

    # layer -> [n_positions, dim] tensor of hidden states
    activations: dict[int, torch.Tensor]
    # Absolute token positions of each captured activation (len = n_positions)
    positions: list[int]
    # Which turn each captured position belongs to (len = n_positions)
    turn_indices: list[int]
    # Stage label for each turn, indexed by turn_idx (len = n_turns)
    stages: list[str]
    # Total tokens in the trajectory (for normalising position into [0, 1])
    total_tokens: int
    # Final test-suite outcome: True = pass, False = fail
    outcome: bool


# Type alias for a single bucket's data
Bucket = tuple[list[torch.Tensor], list[int]]


def bin_by_token_position(
    records: list[TrajectoryRecord],
    layer: int,
    n_bins: int = 10,
) -> list[Bucket]:
    """Group activations into n_bins by normalised token position.

    Position 0 → bin 0; position (total_tokens - 1) → bin (n_bins - 1).
    Each activation is a 1-D tensor of shape [dim].
    Returns a list of n_bins (acts, labels) pairs.
    """
    bins: list[Bucket] = [([], []) for _ in range(n_bins)]
    for record in records:
        acts = record.activations.get(layer)
        if acts is None:
            continue
        label = int(record.outcome)
        denom = max(record.total_tokens - 1, 1)
        for pos_idx, pos in enumerate(record.positions):
            frac = pos / denom
            bin_idx = min(int(frac * n_bins), n_bins - 1)
            bins[bin_idx][0].append(acts[pos_idx])
            bins[bin_idx][1].append(label)
    return bins


def bin_by_tool_call_index(
    records: list[TrajectoryRecord],
    layer: int,
    n_buckets: int = 10,
) -> list[Bucket]:
    """Group activations into n_buckets by normalised turn index.

    Bucket k contains all token activations captured during turn k (i.e. after
    the (k-1)-th tool response and before the k-th tool response).  All tokens
    from a turn are included, not just the last one.
    Turn 0 → bucket 0; last turn → bucket (n_buckets - 1).
    Returns a list of n_buckets (acts, labels) pairs.
    """
    buckets: list[Bucket] = [([], []) for _ in range(n_buckets)]
    for record in records:
        acts = record.activations.get(layer)
        if acts is None or not record.turn_indices:
            continue
        n_turns = max(record.turn_indices) + 1
        label = int(record.outcome)
        denom = max(n_turns - 1, 1)
        for turn_idx in range(n_turns):
            idxs = [i for i, t in enumerate(record.turn_indices) if t == turn_idx]
            if not idxs:
                continue
            frac = turn_idx / denom
            bucket = min(int(frac * n_buckets), n_buckets - 1)
            for i in idxs:
                buckets[bucket][0].append(acts[i])
                buckets[bucket][1].append(label)
    return buckets


def pool_by_stage(
    records: list[TrajectoryRecord],
    layer: int,
    min_samples: int = 10,
) -> dict[str, Bucket]:
    """Group activations by tool-call stage.

    All token activations captured during a turn are included (not just the
    last one), labelled with that turn's stage.
    Stages with fewer than min_samples activations across all records are
    excluded (they would produce unreliable probe estimates).

    Returns a dict mapping stage string → (acts, labels) pair.
    """
    pools: dict[str, Bucket] = {}
    for record in records:
        acts = record.activations.get(layer)
        if acts is None or not record.turn_indices:
            continue
        n_turns = max(record.turn_indices) + 1
        label = int(record.outcome)
        for turn_idx in range(n_turns):
            if turn_idx >= len(record.stages):
                continue
            stage = record.stages[turn_idx]
            idxs = [i for i, t in enumerate(record.turn_indices) if t == turn_idx]
            if not idxs:
                continue
            if stage not in pools:
                pools[stage] = ([], [])
            for i in idxs:
                pools[stage][0].append(acts[i])
                pools[stage][1].append(label)

    return {
        stage: data
        for stage, data in pools.items()
        if len(data[0]) >= min_samples
    }
