"""Tests for interp/swerl/analysis/probe_axes.py.

Spec: three functions partition a list of TrajectoryRecords into
(activations, labels) buckets along different x-axes. All three return
the same uniform shape so the probe training loop can call them uniformly.
"""

import torch
import pytest

from interp.swerl.analysis.probe_axes import (
    TrajectoryRecord,
    bin_by_token_position,
    bin_by_tool_call_index,
    pool_by_stage,
)

DIM = 32
LAYER = 0


def _make_record(
    outcome: bool,
    n_turns: int = 4,
    captures_per_turn: int = 3,
    stages: list[str] | None = None,
    total_tokens: int | None = None,
) -> TrajectoryRecord:
    """Build a synthetic TrajectoryRecord with evenly-spaced captures."""
    n_pos = n_turns * captures_per_turn
    if total_tokens is None:
        total_tokens = n_pos * 10  # positions spread across the trajectory

    torch.manual_seed(int(outcome) * 7 + n_turns)
    activations = {LAYER: torch.randn(n_pos, DIM)}

    positions = [i * 10 for i in range(n_pos)]
    turn_indices = [i // captures_per_turn for i in range(n_pos)]
    if stages is None:
        stages = ["exploration"] * n_turns

    return TrajectoryRecord(
        activations=activations,
        positions=positions,
        turn_indices=turn_indices,
        stages=stages,
        total_tokens=total_tokens,
        outcome=outcome,
    )


# ---------------------------------------------------------------------------
# bin_by_token_position
# ---------------------------------------------------------------------------


def test_bin_by_token_position_returns_n_bins():
    records = [_make_record(True), _make_record(False)]
    result = bin_by_token_position(records, layer=LAYER, n_bins=5)
    assert len(result) == 5


def test_bin_by_token_position_all_data_present():
    """No captures are dropped — every position appears in exactly one bin."""
    records = [_make_record(True, n_turns=2, captures_per_turn=5)]
    n_total = 2 * 5
    result = bin_by_token_position(records, layer=LAYER, n_bins=10)
    total_in_bins = sum(len(acts) for acts, _ in result)
    assert total_in_bins == n_total


def test_bin_by_token_position_labels_match_trajectory_outcome():
    """Every activation in every bin gets the correct trajectory-level label."""
    for outcome in [True, False]:
        record = _make_record(outcome)
        result = bin_by_token_position([record], layer=LAYER, n_bins=10)
        expected_label = int(outcome)
        for acts, labels in result:
            assert all(l == expected_label for l in labels), \
                f"label mismatch for outcome={outcome}"


def test_bin_by_token_position_first_position_in_bin_0():
    """The first captured token (position 0) must land in bin 0."""
    record = _make_record(True, n_turns=1, captures_per_turn=1, total_tokens=100)
    record.positions[0] = 0
    result = bin_by_token_position([record], layer=LAYER, n_bins=10)
    assert len(result[0][0]) > 0, "first position should be in bin 0"


def test_bin_by_token_position_last_position_in_last_bin():
    """The last captured token (at total_tokens - 1) must land in the last bin."""
    total = 100
    record = _make_record(True, n_turns=1, captures_per_turn=1, total_tokens=total)
    record.positions[0] = total - 1
    result = bin_by_token_position([record], layer=LAYER, n_bins=10)
    assert len(result[-1][0]) > 0, "last position should be in the last bin"


# ---------------------------------------------------------------------------
# bin_by_tool_call_index
# ---------------------------------------------------------------------------


def test_bin_by_tool_call_index_returns_n_buckets():
    records = [_make_record(True, n_turns=6), _make_record(False, n_turns=6)]
    result = bin_by_tool_call_index(records, layer=LAYER, n_buckets=6)
    assert len(result) == 6


def test_bin_by_tool_call_index_labels_match_outcome():
    for outcome in [True, False]:
        record = _make_record(outcome, n_turns=4)
        result = bin_by_tool_call_index([record], layer=LAYER, n_buckets=4)
        expected = int(outcome)
        for acts, labels in result:
            assert all(l == expected for l in labels)


def test_bin_by_tool_call_index_uses_last_capture_per_turn():
    """Only the last captured activation per turn contributes to the bucket."""
    record = _make_record(True, n_turns=2, captures_per_turn=3)
    result = bin_by_tool_call_index([record], layer=LAYER, n_buckets=2)
    # 2 turns → 2 non-empty buckets with 1 activation each
    non_empty = [(acts, lbls) for acts, lbls in result if acts]
    assert len(non_empty) == 2
    for acts, _ in non_empty:
        assert len(acts) == 1


# ---------------------------------------------------------------------------
# pool_by_stage
# ---------------------------------------------------------------------------


def test_pool_by_stage_groups_distinct_stages():
    stages = ["exploration", "testing", "editing"]
    record = _make_record(True, n_turns=3, stages=stages)
    result = pool_by_stage([record], layer=LAYER, min_samples=1)
    assert set(result.keys()) == {"exploration", "testing", "editing"}


def test_pool_by_stage_drops_stages_below_min_samples():
    """A stage that appears in fewer trajectories than min_samples is excluded."""
    # Only 1 turn of "testing" across all records
    record = _make_record(True, n_turns=1, stages=["testing"])
    result = pool_by_stage([record], layer=LAYER, min_samples=2)
    assert "testing" not in result


def test_pool_by_stage_keeps_stages_meeting_threshold():
    # 2 records × 1 turn of "testing" = 2 samples → meets min_samples=2
    records = [
        _make_record(True, n_turns=1, stages=["testing"]),
        _make_record(False, n_turns=1, stages=["testing"]),
    ]
    result = pool_by_stage(records, layer=LAYER, min_samples=2)
    assert "testing" in result


def test_pool_by_stage_labels_match_trajectory_outcome():
    records = [
        _make_record(True, n_turns=2, stages=["exploration", "testing"]),
        _make_record(False, n_turns=2, stages=["exploration", "testing"]),
    ]
    result = pool_by_stage(records, layer=LAYER, min_samples=1)
    acts_exp, labels_exp = result["exploration"]
    # One activation per trajectory (last capture per turn) × 2 trajectories
    assert labels_exp == [1, 0] or labels_exp == [0, 1]  # order may vary


# ---------------------------------------------------------------------------
# Uniform return shape across all three functions
# ---------------------------------------------------------------------------


def test_all_axis_functions_return_list_of_acts_labels_tuples():
    """
    The three axis functions all return the same shape so the probe training
    loop can call them uniformly: list of (list[Tensor], list[int]) pairs.
    """
    records = [_make_record(True), _make_record(False)]
    functions_and_results = [
        bin_by_token_position(records, layer=LAYER, n_bins=4),
        bin_by_tool_call_index(records, layer=LAYER, n_buckets=4),
        list(pool_by_stage(records, layer=LAYER, min_samples=1).values()),
    ]
    for result in functions_and_results:
        for bucket in result:
            acts, labels = bucket
            assert isinstance(acts, list)
            assert isinstance(labels, list)
            assert len(acts) == len(labels)
            for a in acts:
                assert isinstance(a, torch.Tensor)
                assert a.shape == (DIM,)
