"""Contrast-Consistent Search (CCS) for program representations.

Finds a direction d in activation space such that for each (original, buggy) pair:
  d · h(original) = -d · h(buggy)

This is the CCS approach from Burns et al. 2022
("Discovering Latent Knowledge in Language Models Without Supervision").

The direction found is label-free: it only requires pairs of opposite stimuli.
It should be more robust than simple mean difference.

Loss: L(d) = E[(d·h+ + d·h-)^2] + E[(d·h+ - d·h- - 1)^2]
  where h+ = original, h- = buggy

Usage:
    python -m interp.bug_trace.analysis.ccs \\
        traj_dir=./interp-bug-trajectories \\
        layer=32
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from cwm.common.params import load_from_cli

logger = logging.getLogger(__name__)


@dataclass
class CCSArgs:
    traj_dir: str = "interp-bug-trajectories"
    layers: list[int] = field(default_factory=lambda: [16, 32, 48, 63])
    time_bin: str = "mean"     # mean | first_quarter | last_quarter | all
    epochs: int = 200
    lr: float = 1e-3
    seed: int = 42
    n_directions: int = 3      # find top N CCS directions (random restarts)


def _get_bin_repr(h_traj: torch.Tensor, time_bin: str) -> torch.Tensor:
    """Extract a single representation vector from a trajectory tensor [T, dim]."""
    T = h_traj.shape[0]
    if T == 0:
        return h_traj.new_zeros(h_traj.shape[1])
    if time_bin == "mean":
        return h_traj.float().mean(0)
    elif time_bin == "first_quarter":
        end = max(1, T // 4)
        return h_traj[:end].float().mean(0)
    elif time_bin == "last_quarter":
        start = max(0, 3 * T // 4)
        return h_traj[start:].float().mean(0)
    elif time_bin == "all":
        return h_traj.float().mean(0)
    else:
        return h_traj.float().mean(0)


def _ccs_loss(d: torch.Tensor, h_pos: torch.Tensor, h_neg: torch.Tensor) -> torch.Tensor:
    """CCS loss for a single direction d [dim].

    h_pos: [N, dim] (original)
    h_neg: [N, dim] (buggy)
    """
    p = (h_pos @ d)  # [N]
    n = (h_neg @ d)  # [N]
    consistency = (p + n).pow(2).mean()
    confidence = (p - n - 1.0).pow(2).mean()
    return consistency + confidence


def _fit_ccs_direction(
    h_pos: torch.Tensor,
    h_neg: torch.Tensor,
    epochs: int,
    lr: float,
    seed: int,
) -> tuple[torch.Tensor, float]:
    """Fit a single CCS direction. Returns (direction [dim], final_loss)."""
    torch.manual_seed(seed)
    dim = h_pos.shape[1]
    d = torch.randn(dim, requires_grad=True)

    opt = torch.optim.Adam([d], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        # Normalise d to unit sphere
        d_n = d / (d.norm() + 1e-8)
        loss = _ccs_loss(d_n, h_pos, h_neg)
        loss.backward()
        opt.step()

    with torch.no_grad():
        d_final = d / (d.norm() + 1e-8)
        final_loss = float(_ccs_loss(d_final, h_pos, h_neg).item())

    return d_final.detach(), final_loss


def run_ccs(args: CCSArgs) -> None:
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(args.seed)

    traj_path = Path(args.traj_dir)
    index: list[dict] = []
    with (traj_path / "index.jsonl").open() as f:
        for line in f:
            index.append(json.loads(line))
    traj_subdir = traj_path / "trajectories"

    # Build paired (original, buggy) matrices per layer
    # Index by original_id to match pairs
    orig_by_id: dict[str, dict] = {}
    buggy_by_id: dict[str, list[dict]] = {}

    for meta in index:
        oid = meta.get("original_id", "")
        if not meta.get("is_buggy", False):
            orig_by_id[oid] = meta
        else:
            buggy_by_id.setdefault(oid, []).append(meta)

    results: dict = {}

    for layer in args.layers:
        h_pos_list, h_neg_list = [], []

        for oid, orig_meta in orig_by_id.items():
            buggy_metas = buggy_by_id.get(oid, [])
            if not buggy_metas:
                continue

            # Load original trajectory
            orig_pt = traj_subdir / f"{orig_meta['sample_id']}.pt"
            if not orig_pt.exists():
                continue
            orig_sample = torch.load(orig_pt, map_location="cpu", weights_only=False)
            orig_traj = orig_sample.get("trajectory", {})
            h_orig = orig_traj.get(layer)
            if h_orig is None:
                h_orig = orig_traj.get(str(layer))
            if h_orig is None:
                continue
            repr_orig = _get_bin_repr(h_orig, args.time_bin)

            for buggy_meta in buggy_metas:
                buggy_pt = traj_subdir / f"{buggy_meta['sample_id']}.pt"
                if not buggy_pt.exists():
                    continue
                buggy_sample = torch.load(buggy_pt, map_location="cpu", weights_only=False)
                buggy_traj = buggy_sample.get("trajectory", {})
                h_buggy = buggy_traj.get(layer)
                if h_buggy is None:
                    h_buggy = buggy_traj.get(str(layer))
                if h_buggy is None:
                    continue
                repr_buggy = _get_bin_repr(h_buggy, args.time_bin)
                h_pos_list.append(repr_orig)
                h_neg_list.append(repr_buggy)

        if len(h_pos_list) < 5:
            logger.warning(f"Layer {layer}: only {len(h_pos_list)} pairs, skipping CCS")
            continue

        h_pos = torch.stack(h_pos_list)
        h_neg = torch.stack(h_neg_list)

        # Normalise
        mu = (h_pos.mean(0) + h_neg.mean(0)) / 2
        std = torch.cat([h_pos, h_neg]).std(0).clamp(min=1e-8)
        h_pos_n = (h_pos - mu) / std
        h_neg_n = (h_neg - mu) / std

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        h_pos_n = h_pos_n.to(device)
        h_neg_n = h_neg_n.to(device)

        # Multiple random restarts
        best_dir, best_loss = None, float("inf")
        all_dirs = []
        for k in range(args.n_directions):
            d, loss = _fit_ccs_direction(h_pos_n, h_neg_n, args.epochs, args.lr, args.seed + k)
            all_dirs.append((d.cpu(), loss))
            if loss < best_loss:
                best_loss = loss
                best_dir = d.cpu()

        # Evaluate: how well does the best direction separate?
        with torch.no_grad():
            d_gpu = best_dir.to(device)
            p_scores = (h_pos_n @ d_gpu).cpu()
            n_scores = (h_neg_n @ d_gpu).cpu()
            # Accuracy: do we correctly predict original > buggy?
            acc = float((p_scores > n_scores).float().mean().item())

        logger.info(
            f"Layer {layer}: CCS loss={best_loss:.4f}, "
            f"separation acc={acc:.3f} (n={h_pos.shape[0]} pairs)"
        )

        results[layer] = {
            "direction": best_dir,
            "loss": best_loss,
            "separation_acc": acc,
            "n_pairs": h_pos.shape[0],
            "feature_mean": mu.cpu(),
            "feature_std": std.cpu(),
            "all_directions": [(d, l) for d, l in all_dirs],
        }
        print(f"  Layer {layer:2d}: CCS acc={acc:.3f}, loss={best_loss:.4f} (n={h_pos.shape[0]})")

    out = traj_path / f"ccs_{args.time_bin}.pt"
    torch.save({"results": results, "args": vars(args)}, out)
    logger.info(f"Saved CCS results to {out}")


if __name__ == "__main__":
    args = load_from_cli(CCSArgs)
    run_ccs(args)
