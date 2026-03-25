"""Representational Similarity Analysis (RSA).

Computes pairwise representational similarity matrices (RSMs) and compares
them to program similarity matrices to assess whether the model's geometry
reflects program structure.

- Model RSM: cosine similarity between trajectory representations
- Program RSM: edit-distance-based similarity between program texts

A high rank correlation (Spearman) between model RSM and program RSM suggests
the model has learned program structure in its representations.

Usage:
    python -m interp.bug_trace.analysis.rsa \\
        traj_dir=./interp-bug-trajectories \\
        layer=32
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch

from cwm.common.params import load_from_cli

logger = logging.getLogger(__name__)


@dataclass
class RSAArgs:
    traj_dir: str = "interp-bug-trajectories"
    layers: list[int] = field(default_factory=lambda: [16, 32, 48, 63])
    time_bin: str = "mean"       # mean | last_quarter
    max_samples: int = 200       # RSM is O(N^2), keep N manageable
    seed: int = 42


def _edit_distance_norm(a: str, b: str) -> float:
    """Normalised character-level edit distance in [0, 1]."""
    import difflib
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    return 1.0 - ratio  # distance (0=identical, 1=completely different)


def _spearman_rho(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman rank correlation between two flat tensors."""
    n = a.shape[0]
    rank_a = a.argsort().argsort().float()
    rank_b = b.argsort().argsort().float()
    rank_a -= rank_a.mean()
    rank_b -= rank_b.mean()
    num = (rank_a * rank_b).sum()
    den = (rank_a.pow(2).sum() * rank_b.pow(2).sum()).sqrt()
    return float((num / den.clamp(min=1e-8)).item())


def run_rsa(args: RSAArgs) -> None:
    logging.basicConfig(level=logging.INFO)
    import random

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    traj_path = Path(args.traj_dir)
    index: list[dict] = []
    with (traj_path / "index.jsonl").open() as f:
        for line in f:
            index.append(json.loads(line))
    traj_subdir = traj_path / "trajectories"

    # Subsample for tractability
    chosen = list(range(len(index)))
    if len(chosen) > args.max_samples:
        chosen = rng.sample(chosen, args.max_samples)

    metas = [index[i] for i in chosen]
    codes = []

    all_results: dict = {}

    for layer in args.layers:
        reprs = []
        valid_metas = []
        valid_codes = []

        for meta in metas:
            sid = meta["sample_id"]
            pt = traj_subdir / f"{sid}.pt"
            if not pt.exists():
                continue
            sample = torch.load(pt, map_location="cpu", weights_only=False)
            traj = sample.get("trajectory", {})
            h = traj.get(layer)
            if h is None:
                h = traj.get(str(layer))
            if h is None or h.shape[0] == 0:
                continue

            T = h.shape[0]
            if args.time_bin == "mean":
                r = h.float().mean(0)
            elif args.time_bin == "last_quarter":
                r = h[max(0, 3 * T // 4):].float().mean(0)
            else:
                r = h.float().mean(0)

            reprs.append(r)
            valid_metas.append(meta)
            valid_codes.append(sample.get("code", ""))

        if len(reprs) < 10:
            continue

        R = torch.stack(reprs)  # [N, dim]
        N = R.shape[0]

        # Model RSM: cosine similarity → distance
        R_norm = torch.nn.functional.normalize(R, dim=-1)
        model_sim = R_norm @ R_norm.T  # [N, N]
        model_dist = 1 - model_sim  # distance matrix

        # Program RSM: edit distance
        prog_dist = torch.zeros(N, N)
        for i in range(N):
            for j in range(i + 1, N):
                d = _edit_distance_norm(valid_codes[i], valid_codes[j])
                prog_dist[i, j] = d
                prog_dist[j, i] = d

        # Extract upper triangle (excluding diagonal)
        mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
        model_flat = model_dist[mask]
        prog_flat = prog_dist[mask]

        rho = _spearman_rho(model_flat, prog_flat)

        # Also compute buggy vs original RSM correlation
        is_buggy = torch.tensor([int(m.get("is_buggy", False)) for m in valid_metas])
        same_bug_status = (is_buggy.unsqueeze(0) == is_buggy.unsqueeze(1)).float()
        rho_bug = _spearman_rho(model_flat, same_bug_status[mask])

        logger.info(
            f"Layer {layer}: Spearman ρ(model, edit_dist)={rho:.4f}, "
            f"ρ(model, same_bug_status)={rho_bug:.4f} (N={N})"
        )
        print(f"  Layer {layer:2d}: ρ(edit_dist)={rho:.4f}, ρ(same_bug)={rho_bug:.4f}")

        all_results[layer] = {
            "spearman_edit_dist": rho,
            "spearman_same_bug_status": rho_bug,
            "n_samples": N,
            "model_dist_upper": model_flat.half(),
            "prog_dist_upper": prog_flat.half(),
        }

    out = traj_path / f"rsa_{args.time_bin}.pt"
    torch.save({"results": all_results, "args": vars(args)}, out)
    logger.info(f"Saved RSA results to {out}")


if __name__ == "__main__":
    args = load_from_cli(RSAArgs)
    run_rsa(args)
