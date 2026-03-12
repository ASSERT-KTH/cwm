import json

import pytest
import torch


def pytest_addoption(parser):
    parser.addoption(
        "--checkpoint-dir",
        default=None,
        help="Path to CWM checkpoint directory (required for GPU integration tests)",
    )


@pytest.fixture
def dim():
    return 128


@pytest.fixture
def n_layers():
    return 4


@pytest.fixture
def fake_activations(dim, n_layers):
    """Simulates extracted activations for one sample: {layer: tensor[n_positions, dim]}"""
    n_positions = 10
    torch.manual_seed(0)
    return {
        layer: torch.randn(n_positions, dim)
        for layer in range(n_layers)
    }


@pytest.fixture
def fake_extract_dir(tmp_path, fake_activations):
    """Creates a minimal extract directory with 20 fake samples."""
    act_dir = tmp_path / "activations"
    act_dir.mkdir()
    index = []
    for i in range(20):
        correct = i % 2 == 0  # alternating correct/incorrect
        # Give each group a distinct bias so mean_diff vectors are nonzero
        group_bias = 1.0 if correct else -1.0
        torch.manual_seed(i)
        sample_activations = {
            layer: fake_activations[layer] + group_bias * 0.5 + torch.randn_like(fake_activations[layer]) * 0.01
            for layer in fake_activations
        }
        sample = {
            "sample_id": str(i),
            "code": f"def f(x):\n    return x + {i}\n",
            "input": "5",
            "output": str(5 + i),
            "mode": "trace_full",
            "generated_text": (
                f"<|return_sep|><|action_sep|> return f(5)"
                f"<|arg_sep|>\"{5 + i}\"<|frame_sep|>"
            ),
            "extracted_answer": str(5 + i),
            "correct": correct,
            "token_ids": [100, 101, 102, 106, 100] * 2,  # fake trace tokens
            "captured_positions": list(range(10)),
            "activations": sample_activations,
        }
        torch.save(sample, act_dir / f"{i}.pt")
        index.append({k: v for k, v in sample.items() if k != "activations"})

    with open(tmp_path / "index.jsonl", "w") as f:
        for entry in index:
            f.write(json.dumps(entry) + "\n")

    return tmp_path
