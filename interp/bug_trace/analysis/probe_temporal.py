"""Train linear probes on decode-step trajectories and produce T×L accuracy heatmaps.

For each (layer L, relative time bin B), extracts h_T for steps T in bin B,
trains a linear probe to predict `is_buggy` / `will_be_correct`, and records val accuracy.

The result is a 2D heatmap: rows = layer, columns = time bin.

Usage:
    python -m interp.bug_trace.analysis.probe_temporal \\
        traj_dir=./interp-bug-trajectories \\
        target=is_buggy
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from cwm.common.params import load_from_cli

logger = logging.getLogger(__name__)


@dataclass
class ProbeTemporalArgs:
    traj_dir: str = "interp-bug-trajectories"
    target: str = "is_buggy"           # is_buggy | will_be_correct
    layers: list[int] = field(default_factory=lambda: [16, 32, 48, 63])
    n_time_bins: int = 10              # split trajectory into N equal bins
    epochs: int = 30
    lr: float = 1e-3
    batch_size: int = 512
    val_fraction: float = 0.2
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
    """Train a linear probe and return val accuracy + probe weights."""
    if X.shape[0] < 10:
        return {"val_acc": float("nan"), "n_train": 0, "n_val": 0}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X.float().to(device)
    y = y.to(device)

    # Mean-center features
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
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        for xb, yb in loader:
            opt.zero_grad()
            crit(probe(xb), yb).backward()
            opt.step()

    probe.eval()
    with torch.no_grad():
        Xv, yv = zip(*[val_ds[i] for i in range(len(val_ds))])
        Xv = torch.stack(Xv)
        yv = torch.stack(yv)
        preds = probe(Xv).argmax(1)
        val_acc = (preds == yv).float().mean().item()

    return {
        "val_acc": val_acc,
        "n_train": n_train,
        "n_val": n_val,
        "probe_weight": probe.weight.detach().cpu(),  # [n_classes, dim]
        "probe_bias": probe.bias.detach().cpu(),
        "feature_mean": mu.cpu(),
    }


def run_probe_temporal(args: ProbeTemporalArgs) -> None:
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(args.seed)

    traj_path = Path(args.traj_dir)
    index_path = traj_path / "index.jsonl"
    index: list[dict] = []
    with index_path.open() as f:
        for line in f:
            index.append(json.loads(line))

    traj_subdir = traj_path / "trajectories"

    # Collect all trajectories and labels
    # group by time bin (relative position in trajectory)
    # data[layer][bin] = list of (h_vector, label)

    label_fn = {
        "is_buggy": lambda meta: int(meta.get("is_buggy", False)),
        "will_be_correct": lambda meta: int(meta.get("correct", False)),
    }.get(args.target)
    if label_fn is None:
        raise ValueError(f"Unknown target: {args.target!r}")

    # Single pass over files, accumulate per-layer in fp16 to save memory.
    # layer_h[L]       : list of [dim] fp16 tensors (one per step, all samples)
    # layer_labels[L]  : list of int labels
    # layer_relpos[L]  : list of float relative positions
    heatmap: dict = {}  # layer → {bin → val_acc}

    layer_h: dict[int, list] = {L: [] for L in args.layers}
    layer_labels: dict[int, list] = {L: [] for L in args.layers}
    layer_relpos: dict[int, list] = {L: [] for L in args.layers}

    for meta in index:
        sid = meta["sample_id"]
        pt = traj_subdir / f"{sid}.pt"
        if not pt.exists():
            continue
        sample = torch.load(pt, map_location="cpu", weights_only=False)
        traj = sample.get("trajectory", {})
        label = label_fn(meta)

        for layer in args.layers:
            h_traj = traj.get(layer)
            if h_traj is None:
                h_traj = traj.get(str(layer))
            if h_traj is None:
                continue
            T = h_traj.shape[0]
            for t in range(T):
                layer_h[layer].append(h_traj[t])   # fp16 [dim]
                layer_labels[layer].append(label)
                layer_relpos[layer].append(t / max(T - 1, 1))

    for layer in args.layers:
        if not layer_h[layer]:
            continue

        H_all = torch.stack(layer_h[layer])            # [N_total, dim] fp16
        y_all = torch.tensor(layer_labels[layer], dtype=torch.long)
        rp_all = torch.tensor(layer_relpos[layer])
        # Free the lists to reclaim memory
        layer_h[layer] = []
        layer_labels[layer] = []
        layer_relpos[layer] = []

        heatmap[layer] = {}
        n_classes = 2

        for bin_idx in range(args.n_time_bins):
            lo = bin_idx / args.n_time_bins
            hi = (bin_idx + 1) / args.n_time_bins
            mask = (rp_all >= lo) & (rp_all < hi)
            if not mask.any():
                continue
            X = H_all[mask].float()   # fp32 only for this bin
            y = y_all[mask]
            probe_result = _train_linear_probe(
                X, y, n_classes, args.epochs, args.lr, args.batch_size,
                args.val_fraction, args.seed,
            )
            heatmap[layer][bin_idx] = probe_result
            logger.info(
                f"L={layer} bin={bin_idx}/{args.n_time_bins} "
                f"val_acc={probe_result['val_acc']:.3f} (n={X.shape[0]})"
            )

    out = traj_path / f"probe_temporal_{args.target}.pt"
    torch.save({"heatmap": heatmap, "args": vars(args)}, out)
    logger.info(f"Saved probe heatmap to {out}")

    # Print summary heatmap
    print(f"\n=== Probe Temporal Heatmap: {args.target} ===")
    print(f"{'Layer':>6}", end="")
    for b in range(args.n_time_bins):
        print(f"  bin{b:02d}", end="")
    print()
    for layer in args.layers:
        print(f"{layer:>6}", end="")
        for b in range(args.n_time_bins):
            val = heatmap.get(layer, {}).get(b, {}).get("val_acc", float("nan"))
            if val != val:
                print("     nan", end="")
            else:
                print(f"  {val:.3f}", end="")
        print()


if __name__ == "__main__":
    args = load_from_cli(ProbeTemporalArgs)
    run_probe_temporal(args)
