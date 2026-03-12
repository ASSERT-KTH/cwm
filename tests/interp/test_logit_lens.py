"""Tests for logit lens (unembedding at intermediate layers)."""

import torch
import pytest


def test_logit_lens_output_shape(dim):
    vocab_size = 1000
    norm = torch.nn.RMSNorm(dim)
    output = torch.nn.Linear(dim, vocab_size, bias=False)
    h = torch.randn(10, dim)
    logits = output(norm(h))
    assert logits.shape == (10, vocab_size)


def test_logit_lens_top_k():
    logits = torch.randn(1, 500)
    top_k = 5
    values, indices = logits.topk(top_k, dim=-1)
    assert indices.shape == (1, top_k)
    # values should be sorted descending
    assert (values[0, :-1] >= values[0, 1:]).all()


def test_logit_lens_softmax_sums_to_one(dim):
    vocab_size = 1000
    norm = torch.nn.RMSNorm(dim)
    output = torch.nn.Linear(dim, vocab_size, bias=False)
    h = torch.randn(1, dim)
    logits = output(norm(h))
    probs = torch.softmax(logits, dim=-1)
    assert abs(probs.sum().item() - 1.0) < 1e-5


def test_logit_lens_rms_norm_custom(dim):
    """Manual RMSNorm matches nn.RMSNorm."""
    norm_weight = torch.ones(dim)
    eps = 1e-5
    h = torch.randn(4, dim)

    # Manual
    rms = h.pow(2).mean(-1, keepdim=True).add(eps).sqrt()
    h_manual = h / rms * norm_weight

    # nn.RMSNorm
    nn_norm = torch.nn.RMSNorm(dim, eps=eps)
    with torch.no_grad():
        nn_norm.weight.fill_(1.0)
        h_nn = nn_norm(h)

    assert torch.allclose(h_manual, h_nn, atol=1e-5)


def test_logit_lens_top1_is_max(dim):
    vocab_size = 200
    logits = torch.randn(3, vocab_size)
    probs = torch.softmax(logits, dim=-1)
    top1_vals, top1_ids = probs.topk(1, dim=-1)
    max_probs = probs.max(-1).values
    assert torch.allclose(top1_vals.squeeze(-1), max_probs)


def test_logit_lens_apply_helper(dim):
    """Test the apply_logit_lens helper function."""
    from interp.logit_lens.run_logit_lens import apply_logit_lens

    vocab_size = 500
    norm_weight = torch.ones(dim)
    output_weight = torch.randn(vocab_size, dim)
    h = torch.randn(5, dim)

    result = apply_logit_lens(h, norm_weight, output_weight, eps=1e-5, top_k=3)
    assert "top_k_ids" in result
    assert "top_k_probs" in result
    assert len(result["top_k_ids"]) == 5
    assert len(result["top_k_ids"][0]) == 3
    # Each position's probs sum to roughly top-k slice, all positive
    for row in result["top_k_probs"]:
        assert all(p >= 0 for p in row)
        assert len(row) == 3
