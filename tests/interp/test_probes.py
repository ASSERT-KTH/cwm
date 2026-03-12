"""Tests for probe model shapes and training step."""

import torch
import pytest

from interp.probes.models import LinearProbe, MLPProbe
from interp.probes.dataset import ProbeDataset


def test_linear_probe_shape(dim):
    probe = LinearProbe(in_dim=dim, n_classes=7)
    x = torch.randn(32, dim)
    out = probe(x)
    assert out.shape == (32, 7)


def test_mlp_probe_shape(dim):
    probe = MLPProbe(in_dim=dim, hidden_dim=64, n_classes=4, n_hidden=1)
    x = torch.randn(32, dim)
    out = probe(x)
    assert out.shape == (32, 4)


def test_mlp2_probe_shape(dim):
    probe = MLPProbe(in_dim=dim, hidden_dim=64, n_classes=2, n_hidden=2)
    x = torch.randn(16, dim)
    out = probe(x)
    assert out.shape == (16, 2)


def test_linear_probe_output_not_all_zero(dim):
    torch.manual_seed(1)
    probe = LinearProbe(in_dim=dim, n_classes=3)
    x = torch.randn(8, dim)
    out = probe(x)
    assert out.abs().sum() > 0


def test_probe_training_step(dim):
    probe = LinearProbe(in_dim=dim, n_classes=2)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-2)
    x = torch.randn(64, dim)
    y = torch.randint(0, 2, (64,))
    loss = torch.nn.functional.cross_entropy(probe(x), y)
    loss.backward()
    optimizer.step()
    assert loss.item() > 0  # finite and positive


def test_probe_dataset_loads(fake_extract_dir, dim):
    ds = ProbeDataset(
        extract_dir=str(fake_extract_dir),
        layer=0,
        target_property="will_be_correct",
    )
    assert len(ds) > 0
    x, y = ds[0]
    assert x.shape[-1] == dim
    assert y in (0, 1)


def test_probe_dataset_layers(fake_extract_dir, n_layers):
    for layer in range(n_layers):
        ds = ProbeDataset(
            extract_dir=str(fake_extract_dir),
            layer=layer,
            target_property="will_be_correct",
        )
        assert len(ds) > 0


def test_probe_dataset_return_type(fake_extract_dir, dim):
    ds = ProbeDataset(
        extract_dir=str(fake_extract_dir),
        layer=0,
        target_property="return_type",
    )
    assert len(ds) > 0
    x, y = ds[0]
    assert x.shape[-1] == dim
    assert isinstance(y, int)


def test_probe_dataset_n_classes(fake_extract_dir):
    ds = ProbeDataset(
        extract_dir=str(fake_extract_dir),
        layer=0,
        target_property="will_be_correct",
    )
    assert ds.n_classes == 2


def test_probe_dataset_missing_layer(fake_extract_dir):
    ds = ProbeDataset(
        extract_dir=str(fake_extract_dir),
        layer=999,  # doesn't exist
        target_property="will_be_correct",
    )
    assert len(ds) == 0
