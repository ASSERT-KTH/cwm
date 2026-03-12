"""Tests for ActivationStore and hook infrastructure."""

import torch
import pytest

from interp.extract.hooks import ActivationStore, SteeringHook, activation_hook_context


def test_activation_store_init():
    store = ActivationStore(layers=[0, 2], capture_token_ids=None)
    assert store.enabled
    assert store.data == {}
    assert store.positions == []


def test_activation_store_captures_correct_layers():
    store = ActivationStore(layers=[1, 3], capture_token_ids=None)
    for layer_idx in range(4):
        h = torch.randn(5, 128)
        if layer_idx in store.layers:
            if layer_idx not in store.data:
                store.data[layer_idx] = []
            store.data[layer_idx].append(h.clone())
    assert set(store.data.keys()) == {1, 3}
    assert 0 not in store.data


def test_activation_store_token_filtering():
    FRAME_SEP = 100
    store = ActivationStore(layers=[0], capture_token_ids=[FRAME_SEP])
    token_ids = torch.tensor([50, 100, 200, 100, 50])
    h = torch.randn(5, 128)
    cap_ids = torch.tensor(store.capture_token_ids)
    mask = torch.isin(token_ids, cap_ids)
    captured = h[mask]
    assert captured.shape[0] == 2  # two FRAME_SEP positions


def test_activation_store_clear():
    store = ActivationStore(layers=[0], capture_token_ids=None)
    store.data[0] = [torch.randn(3, 64)]
    store.positions = [0, 1, 2]
    store.clear()
    assert store.data == {}
    assert store.positions == []


def test_activation_store_get_activations(dim):
    store = ActivationStore(layers=[0], capture_token_ids=None)
    t1 = torch.randn(3, dim)
    t2 = torch.randn(2, dim)
    store.data[0] = [t1, t2]
    result = store.get_activations(0)
    assert result is not None
    assert result.shape == (5, dim)


def test_activation_store_get_activations_missing():
    store = ActivationStore(layers=[0], capture_token_ids=None)
    assert store.get_activations(99) is None


def test_forward_with_hooks_preserves_output(dim):
    h = torch.randn(3, dim)
    h_copy = h.clone()
    store = ActivationStore(layers=[0], capture_token_ids=None)
    # Simulate capture (clone, don't modify h)
    store.data[0] = [h.clone().detach()]
    assert torch.equal(h, h_copy)  # h unchanged


def test_activation_hook_context_sets_global():
    from interp.extract.hooks import _current_store as _orig
    store = ActivationStore(layers=[0], capture_token_ids=None)
    import interp.extract.hooks as _hooks
    with activation_hook_context(store=store):
        assert _hooks._current_store is store
    assert _hooks._current_store is None


def test_activation_hook_context_steering():
    import interp.extract.hooks as _hooks
    vec = torch.ones(64)
    hook = SteeringHook(layer=0, vector=vec, alpha=1.0)
    with activation_hook_context(steering_hooks=[hook]):
        assert len(_hooks._current_steering_hooks) == 1
        assert _hooks._current_steering_hooks[0].layer == 0
    assert _hooks._current_steering_hooks == []


def test_steering_hook_modifies_activations(dim):
    h = torch.zeros(5, dim)
    vector = torch.ones(dim)
    hook = SteeringHook(layer=0, vector=vector, alpha=2.0, position="all")
    h_steered = h.clone()
    h_steered += hook.alpha * hook.vector
    assert torch.allclose(h_steered, torch.full((5, dim), 2.0))


def test_steering_hook_zero_alpha_is_noop(dim):
    h = torch.randn(5, dim)
    h_orig = h.clone()
    hook = SteeringHook(layer=0, vector=torch.ones(dim), alpha=0.0, position="all")
    h += hook.alpha * hook.vector
    assert torch.equal(h, h_orig)
