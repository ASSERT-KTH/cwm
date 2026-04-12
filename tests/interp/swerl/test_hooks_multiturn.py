"""Tests for multi-turn activation accumulation in ActivationStore.

Spec: ActivationStore gains a current_turn counter and a turn_indices list
that are maintained in parallel with positions across multiple g.generate()
calls (one per tool-call turn in a SWEbench trajectory).
"""

import torch
import pytest

from interp.extract.hooks import ActivationStore, _capture_at_layer


# ---------------------------------------------------------------------------
# Turn counter basics
# ---------------------------------------------------------------------------


def test_current_turn_starts_at_zero():
    store = ActivationStore(layers=[0], capture_token_ids=None)
    assert store.current_turn == 0


def test_next_turn_increments_counter():
    store = ActivationStore(layers=[0], capture_token_ids=None)
    store.next_turn()
    assert store.current_turn == 1
    store.next_turn()
    assert store.current_turn == 2


def test_clear_resets_turn_counter():
    store = ActivationStore(layers=[0], capture_token_ids=None)
    store.next_turn()
    store.next_turn()
    store.clear()
    assert store.current_turn == 0


def test_clear_resets_turn_indices():
    store = ActivationStore(layers=[0], capture_token_ids=None)
    h = torch.randn(2, 64)
    q_seqpos = torch.tensor([0, 1])
    _capture_at_layer(store, 0, torch.zeros(2, dtype=torch.long), h, q_seqpos)
    store.clear()
    assert store.turn_indices == []


# ---------------------------------------------------------------------------
# turn_indices tracks which turn each capture came from
# ---------------------------------------------------------------------------


def test_captures_in_turn_zero_get_turn_index_zero():
    store = ActivationStore(layers=[0], capture_token_ids=None)
    h = torch.randn(3, 64)
    q_seqpos = torch.tensor([0, 1, 2])
    _capture_at_layer(store, 0, torch.zeros(3, dtype=torch.long), h, q_seqpos)
    assert store.turn_indices == [0, 0, 0]


def test_captures_across_two_turns_get_distinct_turn_indices():
    """Turn 0 captures and turn 1 captures are distinguishable by turn_indices."""
    store = ActivationStore(layers=[0], capture_token_ids=None)
    h0 = torch.randn(3, 64)
    h1 = torch.randn(2, 64)
    q0 = torch.tensor([0, 1, 2])
    q1 = torch.tensor([10, 11])

    _capture_at_layer(store, 0, torch.zeros(3, dtype=torch.long), h0, q0)
    store.next_turn()
    _capture_at_layer(store, 0, torch.zeros(2, dtype=torch.long), h1, q1)

    assert store.turn_indices == [0, 0, 0, 1, 1]


def test_turn_indices_parallel_to_positions():
    """len(turn_indices) == len(positions) at all times."""
    store = ActivationStore(layers=[0], capture_token_ids=None)
    for turn in range(3):
        h = torch.randn(4, 64)
        q = torch.arange(turn * 4, (turn + 1) * 4)
        _capture_at_layer(store, 0, torch.zeros(4, dtype=torch.long), h, q)
        store.next_turn()

    assert len(store.turn_indices) == len(store.positions)


def test_three_turns_have_correct_indices():
    store = ActivationStore(layers=[0], capture_token_ids=None)
    for turn in range(3):
        h = torch.randn(2, 64)
        q = torch.tensor([turn * 2, turn * 2 + 1])
        _capture_at_layer(store, 0, torch.zeros(2, dtype=torch.long), h, q)
        if turn < 2:
            store.next_turn()

    assert store.turn_indices == [0, 0, 1, 1, 2, 2]


def test_existing_tests_unaffected_by_multiturn_fields():
    """Existing ActivationStore usage (no next_turn calls) still works correctly."""
    store = ActivationStore(layers=[0, 1], capture_token_ids=None)
    for layer in [0, 1]:
        h = torch.randn(5, 64)
        q = torch.arange(5)
        _capture_at_layer(store, layer, torch.zeros(5, dtype=torch.long), h, q)

    result = store.get_activations(0)
    assert result is not None
    assert result.shape == (5, 64)
    # All captures at turn 0 since next_turn was never called
    assert all(t == 0 for t in store.turn_indices)
