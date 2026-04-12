"""Train linear probes on SWEbench activation extracts and generate figures.

Loads all .pt files from an extract directory, trains logistic regression probes
on 3 analysis axes, and saves accuracy-vs-x figures.

Usage:
    python -m interp.swerl.analysis.run_probes \\
        extract_dir=interp-swerl-extract \\
        layer=32 \\
        n_bins=10 \\
        min_samples_per_bucket=20 \\
        out_dir=interp-swerl-extract/figures

Axes produced:
  1. token_position  — probe accuracy vs normalised token-position bin
  2. tool_call_index — probe accuracy vs normalised tool-call-index bucket
  3. stage           — probe accuracy vs tool-call stage (exploration/editing/…)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from cwm.common.params import load_from_cli
from interp.swerl.analysis.probe_axes import (
    TrajectoryRecord,
    bin_by_token_position,
    bin_by_tool_call_index,
    pool_by_stage,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ProbeRunArgs:
    extract_dir: str = "interp-swerl-extract"
    layer: int = 32
    n_bins: int = 10
    # Minimum samples per bucket to train a probe
    min_samples_per_bucket: int = 20
    # Fraction of samples held out for validation
    val_fraction: float = 0.2
    # Max captures to keep per trajectory file (random subsample to cap RAM)
    max_captures_per_file: int = 5000
    # Max samples per bucket (subsample for speed if very large)
    max_samples_per_bucket: int = 50_000
    # Logistic regression settings
    lr_max_iter: int = 1000
    lr_C: float = 1.0
    out_dir: str = ""
    seed: int = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_records(
    extract_dir: Path, layer: int, max_captures_per_file: int = 5000, seed: int = 42
) -> list[TrajectoryRecord]:
    """Load all good .pt files and convert to TrajectoryRecord objects."""
    act_dir = extract_dir / "activations"
    if not act_dir.exists():
        raise FileNotFoundError(f"Activations dir not found: {act_dir}")

    pt_files = sorted(act_dir.glob("*.pt"))
    logger.info(f"Found {len(pt_files)} .pt files in {act_dir}")

    records: list[TrajectoryRecord] = []
    n_empty = 0
    for path in pt_files:
        data = torch.load(path, map_location="cpu", weights_only=False)
        acts = data.get("activations", {})
        layer_acts = acts.get(layer)
        if layer_acts is None or layer_acts.numel() == 0:
            n_empty += 1
            continue

        positions = data.get("positions", [])
        turn_indices = data.get("turn_indices", [])

        # Subsample to cap RAM: keep at most max_captures_per_file rows
        n_cap = layer_acts.shape[0]
        if n_cap > max_captures_per_file:
            rng = torch.Generator().manual_seed(seed)
            idx = torch.randperm(n_cap, generator=rng)[:max_captures_per_file].sort().values
            layer_acts = layer_acts[idx]
            positions = [positions[i] for i in idx.tolist()]
            turn_indices = [turn_indices[i] for i in idx.tolist()]

        turn_log = data.get("turn_log", [])
        # Build stages list indexed by turn_idx
        max_turn = max((e["turn_idx"] for e in turn_log), default=-1)
        stages: list[str] = ["unknown"] * (max_turn + 1)
        for entry in turn_log:
            stages[entry["turn_idx"]] = entry.get("stage", "unknown")

        records.append(
            TrajectoryRecord(
                activations={layer: layer_acts.float()},
                positions=positions,
                turn_indices=turn_indices,
                stages=stages,
                total_tokens=data.get("total_tokens", 1),
                outcome=bool(data.get("outcome", False)),
            )
        )

    logger.info(
        f"Loaded {len(records)} good records "
        f"({n_empty} empty / TP-rank-1 files skipped)"
    )
    return records


# ---------------------------------------------------------------------------
# Probe training
# ---------------------------------------------------------------------------


def _train_probe(
    acts: list[torch.Tensor],
    labels: list[int],
    args: ProbeRunArgs,
) -> dict:
    """Train a linear (logistic regression) probe and return accuracy metrics.

    Uses L2-regularised logistic regression via gradient descent on CPU.
    """
    X = torch.stack(acts).float()  # [N, dim]
    y = torch.tensor(labels, dtype=torch.long)  # [N]

    n = len(X)
    counts = torch.bincount(y, minlength=2)
    majority = float(counts.max()) / n

    if n < args.min_samples_per_bucket or (counts == 0).any():
        return {"n": n, "majority": majority, "val_acc": float("nan"), "skipped": True}

    # Subsample if too large
    if n > args.max_samples_per_bucket:
        g = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(n, generator=g)[: args.max_samples_per_bucket]
        X, y = X[perm], y[perm]
        n = len(X)

    # Standardise
    mu = X.mean(0, keepdim=True)
    std = X.std(0, keepdim=True).clamp(min=1e-6)
    X = (X - mu) / std

    # Train/val split
    n_val = max(1, int(n * args.val_fraction))
    n_train = n - n_val
    g = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(
        TensorDataset(X, y), [n_train, n_val], generator=g
    )

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512)

    dim = X.shape[1]
    probe = nn.Linear(dim, 2)
    optimizer = torch.optim.Adam(
        probe.parameters(), lr=1e-2, weight_decay=1.0 / args.lr_C
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for _ in range(args.lr_max_iter // 10):  # epoch count
        probe.train()
        for xb, yb in train_loader:
            logits = probe(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        probe.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = probe(xb).argmax(-1)
                correct += (preds == yb).sum().item()
                total += len(yb)
        val_acc = correct / total if total > 0 else 0.0
        best_val_acc = max(best_val_acc, val_acc)

    return {"n": n, "majority": majority, "val_acc": best_val_acc, "skipped": False}


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

_STAGE_ORDER = ["exploration", "editing", "testing", "other-bash", "submission", "unknown"]


def _plot_axis(
    x_labels: list[str],
    val_accs: list[float],
    majority_accs: list[float],
    n_samples: list[int],
    title: str,
    xlabel: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(max(8, len(x_labels) * 0.7), 5))

    x = np.arange(len(x_labels))
    valid = [not np.isnan(v) for v in val_accs]

    ax.bar(x[valid], [val_accs[i] for i in range(len(x_labels)) if valid[i]],
           alpha=0.75, label="Probe val acc", color="steelblue")
    ax.bar(x[valid], [majority_accs[i] for i in range(len(x_labels)) if valid[i]],
           alpha=0.45, label="Majority baseline", color="gray")

    # Annotate with sample counts
    for i, (xi, ni) in enumerate(zip(x, n_samples)):
        ax.text(xi, 0.02, f"n={ni}", ha="center", va="bottom",
                fontsize=7, color="black", rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved figure: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_probes(args: ProbeRunArgs) -> None:
    extract_dir = Path(args.extract_dir)
    out_dir = Path(args.out_dir) if args.out_dir else extract_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(
        extract_dir, args.layer,
        max_captures_per_file=args.max_captures_per_file,
        seed=args.seed,
    )
    if not records:
        logger.error("No good records found — aborting.")
        return

    n_pass = sum(r.outcome for r in records)
    logger.info(
        f"Records: {len(records)} total, {n_pass} pass / {len(records) - n_pass} fail "
        f"(layer {args.layer})"
    )

    results: dict = {"layer": args.layer, "n_records": len(records), "n_pass": n_pass}

    # ------------------------------------------------------------------
    # Axis 1: token position bins
    # ------------------------------------------------------------------
    logger.info("=== Axis 1: token position bins ===")
    pos_buckets = bin_by_token_position(records, layer=args.layer, n_bins=args.n_bins)
    pos_results: list[dict] = []
    for bin_idx, (acts, labels) in enumerate(pos_buckets):
        r = _train_probe(acts, labels, args)
        r["bin"] = bin_idx
        pos_results.append(r)
        logger.info(
            f"  Bin {bin_idx:2d}: n={r['n']:6d}, "
            f"majority={r['majority']:.3f}, val_acc={r['val_acc']:.3f}"
            + (" [SKIPPED]" if r.get("skipped") else "")
        )

    _plot_axis(
        x_labels=[str(i) for i in range(args.n_bins)],
        val_accs=[r["val_acc"] for r in pos_results],
        majority_accs=[r["majority"] for r in pos_results],
        n_samples=[r["n"] for r in pos_results],
        title=f"Probe accuracy vs token position bin (layer {args.layer})",
        xlabel="Token position bin (0=start, 9=end)",
        out_path=out_dir / "probe_token_position.png",
    )
    results["token_position"] = pos_results

    # ------------------------------------------------------------------
    # Axis 2: tool-call index bins
    # ------------------------------------------------------------------
    logger.info("=== Axis 2: tool-call index bins ===")
    tc_buckets = bin_by_tool_call_index(records, layer=args.layer, n_buckets=args.n_bins)
    tc_results: list[dict] = []
    for bin_idx, (acts, labels) in enumerate(tc_buckets):
        r = _train_probe(acts, labels, args)
        r["bin"] = bin_idx
        tc_results.append(r)
        logger.info(
            f"  Bucket {bin_idx:2d}: n={r['n']:6d}, "
            f"majority={r['majority']:.3f}, val_acc={r['val_acc']:.3f}"
            + (" [SKIPPED]" if r.get("skipped") else "")
        )

    _plot_axis(
        x_labels=[str(i) for i in range(args.n_bins)],
        val_accs=[r["val_acc"] for r in tc_results],
        majority_accs=[r["majority"] for r in tc_results],
        n_samples=[r["n"] for r in tc_results],
        title=f"Probe accuracy vs tool-call index bucket (layer {args.layer})",
        xlabel="Tool-call index bucket (0=first turn, 9=last turn)",
        out_path=out_dir / "probe_tool_call_index.png",
    )
    results["tool_call_index"] = tc_results

    # ------------------------------------------------------------------
    # Axis 3: stage
    # ------------------------------------------------------------------
    logger.info("=== Axis 3: stage ===")
    stage_pools = pool_by_stage(
        records, layer=args.layer, min_samples=args.min_samples_per_bucket
    )
    # Order stages canonically, append any extras alphabetically
    ordered_stages = [s for s in _STAGE_ORDER if s in stage_pools]
    ordered_stages += sorted(s for s in stage_pools if s not in _STAGE_ORDER)

    stage_results: dict[str, dict] = {}
    for stage in ordered_stages:
        acts, labels = stage_pools[stage]
        r = _train_probe(acts, labels, args)
        r["stage"] = stage
        stage_results[stage] = r
        logger.info(
            f"  {stage:15s}: n={r['n']:6d}, "
            f"majority={r['majority']:.3f}, val_acc={r['val_acc']:.3f}"
            + (" [SKIPPED]" if r.get("skipped") else "")
        )

    _plot_axis(
        x_labels=ordered_stages,
        val_accs=[stage_results[s]["val_acc"] for s in ordered_stages],
        majority_accs=[stage_results[s]["majority"] for s in ordered_stages],
        n_samples=[stage_results[s]["n"] for s in ordered_stages],
        title=f"Probe accuracy vs tool-call stage (layer {args.layer})",
        xlabel="Stage",
        out_path=out_dir / "probe_stage.png",
    )
    results["stage"] = stage_results

    # ------------------------------------------------------------------
    # Save JSON summary
    # ------------------------------------------------------------------
    summary_path = out_dir / "probe_results.json"
    with summary_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved results: {summary_path}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = load_from_cli(ProbeRunArgs)
    run_probes(args)


if __name__ == "__main__":
    main()
