"""Tests for steering vector computation and SteeringHook."""

import torch
import pytest

from interp.extract.hooks import SteeringHook
from interp.steering.vectors import compute_steering_vectors


def test_mean_diff_vector_shape(fake_extract_dir, dim, n_layers):
    vectors = compute_steering_vectors(
        extract_dir=str(fake_extract_dir),
        condition="correct_vs_incorrect",
        layers=list(range(n_layers)),
        method="mean_diff",
    )
    assert len(vectors) > 0
    for layer, v in vectors.items():
        assert v.shape == (dim,), f"Layer {layer}: expected ({dim},), got {v.shape}"


def test_mean_diff_vector_normalized(fake_extract_dir, n_layers):
    vectors = compute_steering_vectors(
        extract_dir=str(fake_extract_dir),
        condition="correct_vs_incorrect",
        layers=list(range(n_layers)),
        method="mean_diff",
    )
    for layer, v in vectors.items():
        norm = v.norm().item()
        assert abs(norm - 1.0) < 1e-4, f"Layer {layer}: norm={norm}"


def test_mean_diff_vector_nonzero(fake_extract_dir, n_layers):
    vectors = compute_steering_vectors(
        extract_dir=str(fake_extract_dir),
        condition="correct_vs_incorrect",
        layers=list(range(n_layers)),
        method="mean_diff",
    )
    for layer, v in vectors.items():
        assert v.abs().sum() > 0, f"Layer {layer}: vector is all zeros"


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


def test_steering_hook_last_position(dim):
    h = torch.zeros(5, dim)
    vector = torch.ones(dim)
    hook = SteeringHook(layer=0, vector=vector, alpha=1.0, position="last")
    h[-1:] += hook.alpha * hook.vector
    assert torch.allclose(h[-1], torch.ones(dim))
    assert torch.allclose(h[:-1], torch.zeros(4, dim))


def test_steering_hook_fields(dim):
    vec = torch.randn(dim)
    hook = SteeringHook(layer=5, vector=vec, alpha=0.5, position="trace_tokens")
    assert hook.layer == 5
    assert hook.alpha == 0.5
    assert hook.position == "trace_tokens"
    assert hook.vector is vec


def test_compute_vectors_pca(fake_extract_dir, dim, n_layers):
    vectors = compute_steering_vectors(
        extract_dir=str(fake_extract_dir),
        condition="correct_vs_incorrect",
        layers=list(range(n_layers)),
        method="pca",
    )
    assert len(vectors) > 0
    for layer, v in vectors.items():
        assert v.shape == (dim,)
        norm = v.norm().item()
        assert abs(norm - 1.0) < 1e-4
