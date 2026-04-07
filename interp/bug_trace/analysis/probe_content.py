"""Probe for content-specific program properties: mutation_type (5-class).

Unlike is_buggy (a prompt artifact), mutation_type requires the model to
distinguish WHICH specific change was made — a stronger test of whether the
hidden state encodes program content.

The mutation token IS visible in the prompt code, so early-bin accuracy will
reflect prompt-reading. The key question is the temporal profile:
  - Flat across bins → purely prompt-reading artifact
  - Rising across bins → model actively computes/maintains bug type through reasoning

Includes:
  - Per-(layer, bin) probe accuracy
  - Permutation baseline (shuffled labels, 5 repeats) for significance
  - Per-class prompt-length distribution (confound check)

Usage:
    python -m interp.bug_trace.analysis.probe_content \\
        traj_dir=./interp-bug-trajectories
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from cwm.common.params import load_from_cli

logger = logging.getLogger(__name__)

EASY_MUTATION_TYPES = [
    "condition_flip",
    "off_by_one_minus",
    "off_by_one_plus",
    "wrong_comparator",
    "wrong_operator",
]

HARD_MUTATION_TYPES = [
    "deleted_accumulator",
    "swapped_arguments",
    "wrong_variable",
]

# Default: all known mutation types. The probe will auto-restrict to whichever
# types are actually present in the dataset (see run_probe_content below).
MUTATION_TYPES = EASY_MUTATION_TYPES + HARD_MUTATION_TYPES
MUTATION_TO_IDX = {m: i for i, m in enumerate(MUTATION_TYPES)}


@dataclass
class ProbeContentArgs:
    traj_dir: str = "interp-bug-trajectories"
    layers: list[int] = field(default_factory=lambda: [16, 32, 48, 63])
    n_time_bins: int = 10
    epochs: int = 30
    lr: float = 1e-3
    batch_size: int = 512
    val_fraction: float = 0.2
    n_perm: int = 5        # number of permutation repeats for baseline
    seed: int = 42


def _train_linear_probe(
    X: torch.Tensor,  # [N, dim]
    y: torch.Tensor,  # [N] long
    n_classes: int,
    epochs: int,
    lr: float,
    batch_size: int,
    val_fraction: float,
    seed: int,
) -> dict:
    if X.shape[0] < 10:
        return {"val_acc": float("nan"), "n_train": 0, "n_val": 0}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X.float().to(device)
    y = y.to(device)

    mu = X.mean(0, keepdim=True)
    X = X - mu

    n = X.shape[0]
    n_val = max(1, int(n * val_fraction))
    n_train = n - n_val
    gen = torch.Generator().manual_seed(seed)
    ds = TensorDataset(X, y)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=gen)

    probe = nn.Linear(X.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    for _ in range(epochs):
        probe.train()
        for xb, yb in DataLoader(train_ds, batch_size=batch_size, shuffle=True):
            opt.zero_grad()
            crit(probe(xb), yb).backward()
            opt.step()

    probe.eval()
    with torch.no_grad():
        Xv = torch.stack([val_ds[i][0] for i in range(len(val_ds))])
        yv = torch.stack([val_ds[i][1] for i in range(len(val_ds))])
        val_acc = float((probe(Xv).argmax(1) == yv).float().mean().item())

    return {"val_acc": val_acc, "n_train": n_train, "n_val": n_val}


def _perm_baseline(
    X: torch.Tensor,
    y: torch.Tensor,
    n_classes: int,
    epochs: int,
    lr: float,
    batch_size: int,
    val_fraction: float,
    seed: int,
    n_perm: int,
) -> float:
    """Mean val_acc under shuffled labels (permutation baseline)."""
    accs = []
    for k in range(n_perm):
        perm = torch.randperm(len(y), generator=torch.Generator().manual_seed(seed + k + 1000))
        y_shuf = y[perm]
        res = _train_linear_probe(X, y_shuf, n_classes, epochs, lr, batch_size, val_fraction, seed + k)
        if res["val_acc"] == res["val_acc"]:  # not nan
            accs.append(res["val_acc"])
    return float(sum(accs) / len(accs)) if accs else float("nan")


def run_probe_content(args: ProbeContentArgs) -> None:
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(args.seed)

    traj_path = Path(args.traj_dir)
    index: list[dict] = []
    with (traj_path / "index.jsonl").open() as f:
        for line in f:
            index.append(json.loads(line))

    traj_subdir = traj_path / "trajectories"

    # Determine which mutation types are present in this dataset
    from collections import Counter
    all_types_in_data = Counter(
        m.get("mutation_type", "none")
        for m in index
        if m.get("is_buggy", False)
    )
    present_types = sorted([t for t in all_types_in_data if t in MUTATION_TO_IDX and all_types_in_data[t] > 0])
    if not present_types:
        logger.error("No known mutation types found in dataset. Check mutation_type values.")
        return
    logger.info(f"Mutation types present in dataset: {present_types}")

    # Build a local mapping restricted to present types
    local_types = present_types
    local_to_idx = {m: i for i, m in enumerate(local_types)}

    # Only use buggy samples with known mutation_type
    buggy_index = [
        m for m in index
        if m.get("is_buggy", False) and m.get("mutation_type", "none") in local_to_idx
    ]
    logger.info(f"Buggy samples with known mutation_type: {len(buggy_index)}")

    # Class distribution and majority baseline
    counts = Counter(m["mutation_type"] for m in buggy_index)
    total = sum(counts.values())
    print("\n=== Class distribution (mutation_type) ===")
    for mt in local_types:
        n = counts.get(mt, 0)
        print(f"  {mt:<25} {n:>4} ({100*n/max(total,1):.1f}%)")
    majority_baseline = max(counts.values()) / total if total > 0 else 1.0 / len(local_types)
    random_baseline = 1.0 / len(local_types)
    print(f"  Random baseline:   {random_baseline:.3f}")
    print(f"  Majority baseline: {majority_baseline:.3f}")

    # Prompt length per class (using n_prompt_tokens from .pt file)
    print("\n=== Prompt length by mutation_type (confound check) ===")
    length_by_type: dict[str, list] = {mt: [] for mt in local_types}
    for meta in buggy_index:
        pt = traj_subdir / f"{meta['sample_id']}.pt"
        if pt.exists():
            sample = torch.load(pt, map_location="cpu", weights_only=False)
            n_prompt = sample.get("n_prompt_tokens", 0)
            if n_prompt > 0:
                length_by_type[meta["mutation_type"]].append(n_prompt)
    for mt in local_types:
        ls = length_by_type[mt]
        if ls:
            print(f"  {mt:<25} mean={sum(ls)/len(ls):.0f} tokens (n={len(ls)})")

    n_classes = len(local_types)
    heatmap: dict = {}

    print(f"\n=== Probe heatmap: mutation_type ({n_classes}-class) ===")
    print(f"{'Layer':>6}", end="")
    for b in range(args.n_time_bins):
        print(f"  bin{b:02d}", end="")
    print("  | perm_bl")

    # Process one layer at a time to avoid OOM
    for layer in args.layers:
        h_list: list = []
        label_list: list = []
        relpos_list: list = []

        for meta in buggy_index:
            sid = meta["sample_id"]
            pt = traj_subdir / f"{sid}.pt"
            if not pt.exists():
                continue
            sample = torch.load(pt, map_location="cpu", weights_only=False)
            traj = sample.get("trajectory", {})
            label = local_to_idx[meta["mutation_type"]]
            h_traj = traj.get(layer)
            if h_traj is None:
                h_traj = traj.get(str(layer))
            if h_traj is None:
                continue
            T = h_traj.shape[0]
            for t in range(T):
                h_list.append(h_traj[t])
                label_list.append(label)
                relpos_list.append(t / max(T - 1, 1))

        if not h_list:
            continue

        H_all = torch.stack(h_list)
        y_all = torch.tensor(label_list, dtype=torch.long)
        rp_all = torch.tensor(relpos_list)
        del h_list, label_list, relpos_list

        heatmap[layer] = {}
        row_accs = []

        for bin_idx in range(args.n_time_bins):
            lo = bin_idx / args.n_time_bins
            hi = (bin_idx + 1) / args.n_time_bins
            mask = (rp_all >= lo) & (rp_all < hi)
            if not mask.any():
                heatmap[layer][bin_idx] = {"val_acc": float("nan")}
                row_accs.append(float("nan"))
                continue

            X = H_all[mask].float()
            y = y_all[mask]
            res = _train_linear_probe(
                X, y, n_classes, args.epochs, args.lr, args.batch_size,
                args.val_fraction, args.seed,
            )
            heatmap[layer][bin_idx] = res
            row_accs.append(res["val_acc"])
            logger.info(
                f"L={layer} bin={bin_idx} val_acc={res['val_acc']:.3f} (n={X.shape[0]})"
            )

        # Permutation baseline on all data for this layer
        perm_bl = _perm_baseline(
            H_all.float(), y_all, n_classes, args.epochs, args.lr,
            args.batch_size, args.val_fraction, args.seed, args.n_perm,
        )
        heatmap[layer]["perm_baseline"] = perm_bl
        heatmap[layer]["majority_baseline"] = majority_baseline
        heatmap[layer]["random_baseline"] = random_baseline
        del H_all, y_all, rp_all  # free before next layer pass

        print(f"{layer:>6}", end="")
        for acc in row_accs:
            if acc != acc:
                print("     nan", end="")
            else:
                print(f"  {acc:.3f}", end="")
        print(f"  | {perm_bl:.3f}")

    out = traj_path / "probe_content_mutation_type.pt"
    torch.save({"heatmap": heatmap, "args": vars(args),
                "mutation_types": local_types,
                "class_counts": dict(counts),
                "majority_baseline": majority_baseline,
                "random_baseline": random_baseline}, out)
    logger.info(f"Saved content probe results to {out}")


if __name__ == "__main__":
    args = load_from_cli(ProbeContentArgs)
    run_probe_content(args)
