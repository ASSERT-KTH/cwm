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
import random
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
    test_fraction: float = 0.3  # fraction of original_ids held out for evaluation


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
    orig_by_id: dict[str, dict] = {}
    buggy_by_id: dict[str, list[dict]] = {}

    for meta in index:
        oid = meta.get("original_id", "")
        if not meta.get("is_buggy", False):
            orig_by_id[oid] = meta
        else:
            buggy_by_id.setdefault(oid, []).append(meta)

    # Train/test split by original_id so no program appears in both splits.
    # This ensures the CCS direction is never fitted on test-set samples,
    # making the causal steering evaluation (activation_patch.py) a clean
    # generalization test.
    all_oids = sorted(orig_by_id.keys())
    rng = random.Random(args.seed)
    rng.shuffle(all_oids)
    n_test = max(1, int(len(all_oids) * args.test_fraction))
    test_oids = set(all_oids[:n_test])
    train_oids = set(all_oids[n_test:])

    logger.info(
        f"CCS split: {len(train_oids)} train originals / {len(test_oids)} test originals "
        f"(test_fraction={args.test_fraction})"
    )

    # Count train/test pairs
    n_train_pairs = sum(len(buggy_by_id.get(oid, [])) for oid in train_oids)
    n_test_pairs = sum(len(buggy_by_id.get(oid, [])) for oid in test_oids)
    print(f"  Train pairs: {n_train_pairs}, Test pairs: {n_test_pairs}")

    # Save split for use by activation_patch.py
    split_info = {
        "train_original_ids": sorted(train_oids),
        "test_original_ids": sorted(test_oids),
        "test_fraction": args.test_fraction,
        "seed": args.seed,
        "n_train_originals": len(train_oids),
        "n_test_originals": len(test_oids),
        "n_train_pairs": n_train_pairs,
        "n_test_pairs": n_test_pairs,
    }
    split_path = traj_path / "ccs_split.json"
    with split_path.open("w") as f:
        json.dump(split_info, f, indent=2)
    logger.info(f"Saved CCS split to {split_path}")

    results: dict = {}

    def _load_pairs_for_oids(oid_set, layer):
        """Load (h_pos, h_neg) pairs for a given set of original_ids."""
        pos_list, neg_list = [], []
        for oid in oid_set:
            orig_meta = orig_by_id.get(oid)
            if orig_meta is None:
                continue
            buggy_metas = buggy_by_id.get(oid, [])
            if not buggy_metas:
                continue
            orig_pt = traj_subdir / f"{orig_meta['sample_id']}.pt"
            if not orig_pt.exists():
                continue
            orig_sample = torch.load(orig_pt, map_location="cpu", weights_only=False)
            h_orig = orig_sample.get("trajectory", {}).get(layer)
            if h_orig is None:
                h_orig = orig_sample.get("trajectory", {}).get(str(layer))
            if h_orig is None:
                continue
            repr_orig = _get_bin_repr(h_orig, args.time_bin)
            for buggy_meta in buggy_metas:
                buggy_pt = traj_subdir / f"{buggy_meta['sample_id']}.pt"
                if not buggy_pt.exists():
                    continue
                buggy_sample = torch.load(buggy_pt, map_location="cpu", weights_only=False)
                h_buggy = buggy_sample.get("trajectory", {}).get(layer)
                if h_buggy is None:
                    h_buggy = buggy_sample.get("trajectory", {}).get(str(layer))
                if h_buggy is None:
                    continue
                pos_list.append(repr_orig)
                neg_list.append(_get_bin_repr(h_buggy, args.time_bin))
        return pos_list, neg_list

    for layer in args.layers:
        # Load train pairs (used to fit the CCS direction)
        h_pos_list, h_neg_list = _load_pairs_for_oids(train_oids, layer)

        if len(h_pos_list) < 5:
            logger.warning(f"Layer {layer}: only {len(h_pos_list)} train pairs, skipping CCS")
            continue

        h_pos = torch.stack(h_pos_list)
        h_neg = torch.stack(h_neg_list)

        # Normalise using train-set statistics only
        mu = (h_pos.mean(0) + h_neg.mean(0)) / 2
        std = torch.cat([h_pos, h_neg]).std(0).clamp(min=1e-8)
        h_pos_n = (h_pos - mu) / std
        h_neg_n = (h_neg - mu) / std

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        h_pos_n = h_pos_n.to(device)
        h_neg_n = h_neg_n.to(device)

        # Multiple random restarts, select best direction on TRAIN set
        best_dir, best_loss = None, float("inf")
        all_dirs = []
        for k in range(args.n_directions):
            d, loss = _fit_ccs_direction(h_pos_n, h_neg_n, args.epochs, args.lr, args.seed + k)
            all_dirs.append((d.cpu(), loss))
            if loss < best_loss:
                best_loss = loss
                best_dir = d.cpu()

        # Train-set separation accuracy
        with torch.no_grad():
            d_gpu = best_dir.to(device)
            train_acc = float(
                ((h_pos_n @ d_gpu) > (h_neg_n @ d_gpu)).float().mean().item()
            )

        # Test-set evaluation (held-out pairs — never seen during fitting)
        test_pos_list, test_neg_list = _load_pairs_for_oids(test_oids, layer)
        test_acc = float("nan")
        n_test = 0
        if test_pos_list:
            tp = torch.stack(test_pos_list)
            tn = torch.stack(test_neg_list)
            tp_n = ((tp - mu) / std).to(device)
            tn_n = ((tn - mu) / std).to(device)
            with torch.no_grad():
                test_acc = float(
                    ((tp_n @ d_gpu) > (tn_n @ d_gpu)).float().mean().item()
                )
            n_test = tp.shape[0]

        logger.info(
            f"Layer {layer}: CCS loss={best_loss:.4f}, "
            f"train_acc={train_acc:.3f} (n={h_pos.shape[0]}), "
            f"test_acc={test_acc:.3f} (n={n_test})"
        )

        results[layer] = {
            "direction": best_dir,
            "loss": best_loss,
            "separation_acc": train_acc,       # backward-compat key
            "train_separation_acc": train_acc,
            "test_separation_acc": test_acc,
            "n_pairs": h_pos.shape[0],
            "n_test_pairs": n_test,
            "feature_mean": mu.cpu(),
            "feature_std": std.cpu(),
            "all_directions": [(d, l) for d, l in all_dirs],
        }
        print(
            f"  Layer {layer:2d}: train_acc={train_acc:.3f}  "
            f"test_acc={test_acc:.3f}  loss={best_loss:.4f}  "
            f"(n_train={h_pos.shape[0]}, n_test={n_test})"
        )

    out = traj_path / f"ccs_{args.time_bin}.pt"
    torch.save({"results": results, "args": vars(args)}, out)
    logger.info(f"Saved CCS results to {out}")


if __name__ == "__main__":
    args = load_from_cli(CCSArgs)
    run_ccs(args)
