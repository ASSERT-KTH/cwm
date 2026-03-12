"""Contrastive steering vector computation from extracted activations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import torch


# ---------------------------------------------------------------------------
# Pre-defined condition functions
# ---------------------------------------------------------------------------

def _condition_correct(meta: dict) -> bool | None:
    """True if the sample was correct, False if incorrect, None to skip."""
    return meta.get("correct", None)


def _condition_positive_return(meta: dict) -> bool | None:
    """True if extracted_answer evaluates to a positive number."""
    import ast

    ans = meta.get("extracted_answer")
    if ans is None:
        return None
    try:
        val = ast.literal_eval(ans)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return val > 0
    except Exception:
        pass
    return None


def _condition_truthy_return(meta: dict) -> bool | None:
    """True if extracted_answer is truthy."""
    import ast

    ans = meta.get("extracted_answer")
    if ans is None:
        return None
    try:
        val = ast.literal_eval(ans)
        return bool(val)
    except Exception:
        return None


def _condition_numeric_return(meta: dict) -> bool | None:
    """True for numeric types (int/float), False for string."""
    import ast

    ans = meta.get("extracted_answer")
    if ans is None:
        return None
    try:
        val = ast.literal_eval(ans)
        if isinstance(val, bool):
            return None
        if isinstance(val, (int, float)):
            return True
        if isinstance(val, str):
            return False
    except Exception:
        pass
    return None


_CONDITION_FNS: dict[str, Callable[[dict], bool | None]] = {
    "correct_vs_incorrect": _condition_correct,
    "positive_vs_negative": _condition_positive_return,
    "truthy_vs_falsy": _condition_truthy_return,
    "numeric_vs_string": _condition_numeric_return,
}


# ---------------------------------------------------------------------------
# Vector computation
# ---------------------------------------------------------------------------

def compute_steering_vectors(
    extract_dir: str,
    condition: str | Callable[[dict], bool | None],
    layers: list[int],
    position: str = "last_return",  # last_return | first | mean
    method: str = "mean_diff",      # mean_diff | pca
) -> dict[int, torch.Tensor]:
    """Compute contrastive steering vectors.

    Args:
        extract_dir: Path to Phase-1 output directory.
        condition: Name of a pre-defined condition or a callable
            ``(meta_dict) -> bool | None``.  True = group A,
            False = group B, None = skip.
        layers: Which layers to compute vectors for.
        position: Which activation position to use per sample.
        method: "mean_diff" or "pca".

    Returns:
        {layer_idx: unit-normalized steering vector [dim]}
    """
    extract_path = Path(extract_dir)
    act_dir = extract_path / "activations"

    if isinstance(condition, str):
        condition_fn = _CONDITION_FNS[condition]
    else:
        condition_fn = condition

    # Read index
    index: list[dict] = []
    with (extract_path / "index.jsonl").open() as f:
        for line in f:
            index.append(json.loads(line))

    # Collect per-layer per-group activations
    group_a: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    group_b: dict[int, list[torch.Tensor]] = {l: [] for l in layers}

    for meta in index:
        label = condition_fn(meta)
        if label is None:
            continue

        sid = meta["sample_id"]
        pt_path = act_dir / f"{sid}.pt"
        if not pt_path.exists():
            continue

        sample = torch.load(pt_path, map_location="cpu", weights_only=False)
        activations = sample.get("activations", {})

        for layer in layers:
            if layer not in activations:
                continue
            h = activations[layer]  # [n_pos, dim]
            if h.shape[0] == 0:
                continue

            # Select position
            if position == "last_return":
                # Use last captured position (typically last trace event)
                vec = h[-1]
            elif position == "first":
                vec = h[0]
            elif position == "mean":
                vec = h.mean(0)
            else:
                vec = h[-1]

            if label:
                group_a[layer].append(vec)
            else:
                group_b[layer].append(vec)

    steering_vectors: dict[int, torch.Tensor] = {}

    for layer in layers:
        a_vecs = group_a[layer]
        b_vecs = group_b[layer]

        if not a_vecs or not b_vecs:
            continue

        a_mean = torch.stack(a_vecs).mean(0)
        b_mean = torch.stack(b_vecs).mean(0)

        if method == "mean_diff":
            direction = a_mean - b_mean
        elif method == "pca":
            # First PC of pairwise difference vectors
            n = min(len(a_vecs), len(b_vecs))
            diffs = torch.stack(a_vecs[:n]) - torch.stack(b_vecs[:n])
            # SVD for first PC
            _, _, vh = torch.linalg.svd(diffs - diffs.mean(0), full_matrices=False)
            direction = vh[0]
        else:
            raise ValueError(f"Unknown method: {method!r}")

        norm = direction.norm()
        if norm > 0:
            direction = direction / norm

        steering_vectors[layer] = direction

    return steering_vectors


def save_steering_vectors(
    vectors: dict[int, torch.Tensor],
    path: str,
    condition: str = "",
    method: str = "",
) -> None:
    """Save steering vectors to a .pt file."""
    torch.save(
        {
            "vectors": vectors,
            "condition": condition,
            "method": method,
        },
        path,
    )


def load_steering_vectors(path: str) -> dict[int, torch.Tensor]:
    """Load steering vectors from a .pt file."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data["vectors"]
