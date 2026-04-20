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

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

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
    # W&B
    wandb_project: str = "cwm-interp"
    wandb_run_name: str = ""
    wandb_disabled: bool = False
    # Set to False to skip the global-probe experiment (train once, eval per axis)
    run_global_probe: bool = True
    # Hard cap on total samples fed to the global probe (independent of per-bucket cap)
    max_global_probe_samples: int = 200_000


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


def _save_probe(probe: nn.Linear, mu: torch.Tensor, std: torch.Tensor, path: Path) -> None:
    """Save probe weights and normalization stats to a .pt checkpoint."""
    torch.save({"state_dict": probe.state_dict(), "mu": mu, "std": std}, path)


def _train_probe(
    acts: list[torch.Tensor],
    labels: list[int],
    args: ProbeRunArgs,
    save_path: Path | None = None,
) -> dict:
    """Train a linear (logistic regression) probe and return accuracy metrics.

    Uses L2-regularised logistic regression via gradient descent on CPU.
    If save_path is given, saves the probe checkpoint there.
    """
    n = len(acts)
    y_all = torch.tensor(labels, dtype=torch.long)
    counts = torch.bincount(y_all, minlength=2)
    majority = float(counts.max()) / n

    if n < args.min_samples_per_bucket or (counts == 0).any():
        return {"n": n, "majority": majority, "val_acc": float("nan"), "skipped": True}

    # Subsample the list BEFORE stacking to avoid OOM on large buckets
    if args.max_samples_per_bucket > 0 and n > args.max_samples_per_bucket:
        g = torch.Generator().manual_seed(args.seed)
        idx = torch.randperm(n, generator=g)[: args.max_samples_per_bucket].tolist()
        acts = [acts[i] for i in idx]
        labels = [labels[i] for i in idx]
        n = len(acts)
        counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=2)
        majority = float(counts.max()) / n

    X = torch.stack(acts).float()  # [N, dim]
    y = torch.tensor(labels, dtype=torch.long)  # [N]

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

    if save_path is not None:
        _save_probe(probe, mu, std, save_path)

    return {"n": n, "majority": majority, "val_acc": best_val_acc, "skipped": False}


# ---------------------------------------------------------------------------
# Single-probe experiment
# ---------------------------------------------------------------------------


def _train_global_probe_all_axes(
    records: list[TrajectoryRecord],
    layer: int,
    n_bins: int,
    args: ProbeRunArgs,
    save_path: Path | None = None,
) -> tuple[list[dict], list[dict], dict[str, dict]]:
    """Train ONE probe on all activations, evaluate per axis.

    Collects every captured activation across all records with per-activation
    axis metadata (position fraction, turn fraction, stage).  Performs a single
    global 80/20 train/val split, trains one linear probe, then slices the val
    predictions by each axis to produce per-group accuracy.
    If save_path is given, saves the probe checkpoint there.

    Returns:
        pos_results   — list[dict] indexed 0…n_bins-1  (token position bins)
        tc_results    — list[dict] indexed 0…n_bins-1  (tool-call index buckets)
        stage_results — dict[stage → dict]
    """
    all_acts: list[torch.Tensor] = []
    all_labels: list[int] = []
    all_pos_fracs: list[float] = []
    all_turn_fracs: list[float] = []
    all_stages_list: list[str] = []

    for record in records:
        acts = record.activations.get(layer)
        if acts is None or not record.turn_indices:
            continue
        label = int(record.outcome)
        denom_pos = max(record.total_tokens - 1, 1)
        n_turns = max(record.turn_indices) + 1
        denom_turn = max(n_turns - 1, 1)
        for pos_idx, (pos, turn_idx) in enumerate(zip(record.positions, record.turn_indices)):
            all_acts.append(acts[pos_idx])
            all_labels.append(label)
            all_pos_fracs.append(pos / denom_pos)
            all_turn_fracs.append(turn_idx / denom_turn)
            stage = record.stages[turn_idx] if turn_idx < len(record.stages) else "unknown"
            all_stages_list.append(stage)

    n = len(all_acts)
    nan_result = {"n": 0, "majority": float("nan"), "val_acc": float("nan"), "skipped": True}
    if n == 0:
        return (
            [{**nan_result, "bin": i} for i in range(n_bins)],
            [{**nan_result, "bin": i} for i in range(n_bins)],
            {},
        )

    # Subsample the list BEFORE stacking to avoid OOM
    cap = args.max_global_probe_samples
    if cap > 0 and n > cap:
        g = torch.Generator().manual_seed(args.seed)
        keep = torch.randperm(n, generator=g)[:cap].sort().values.tolist()
        all_acts = [all_acts[i] for i in keep]
        all_labels = [all_labels[i] for i in keep]
        all_pos_fracs = [all_pos_fracs[i] for i in keep]
        all_turn_fracs = [all_turn_fracs[i] for i in keep]
        all_stages_list = [all_stages_list[i] for i in keep]
        n = cap

    X = torch.stack(all_acts).float()
    y = torch.tensor(all_labels, dtype=torch.long)

    # Global train/val split
    n_val = max(1, int(n * args.val_fraction))
    n_train = n - n_val
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n, generator=g)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    mu = X_train.mean(0, keepdim=True)
    std = X_train.std(0, keepdim=True).clamp(min=1e-6)
    X_train = (X_train - mu) / std
    X_val = (X_val - mu) / std

    # Train single probe
    dim = X_train.shape[1]
    probe = nn.Linear(dim, 2)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-2, weight_decay=1.0 / args.lr_C)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=512, shuffle=True)
    for _ in range(args.lr_max_iter // 10):
        probe.train()
        for xb, yb in train_loader:
            loss = criterion(probe(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    probe.eval()
    with torch.no_grad():
        val_preds = probe(X_val).argmax(-1)

    val_idx_list = val_idx.tolist()

    def _eval_per_group(group_ids_val: list) -> dict:
        """Given per-val-sample group ids, return accuracy keyed by group id."""
        unique_groups = sorted(set(group_ids_val))
        out = {}
        for gid in unique_groups:
            mask = torch.tensor([g == gid for g in group_ids_val], dtype=torch.bool)
            n_b = int(mask.sum())
            if n_b == 0:
                out[gid] = {**nan_result}
                continue
            counts = torch.bincount(y_val[mask], minlength=2)
            majority = float(counts.max()) / n_b
            acc = float((val_preds[mask] == y_val[mask]).float().mean())
            out[gid] = {"n": n_b, "majority": majority, "val_acc": acc, "skipped": False}
        return out

    # Axis 1: token position bins
    pos_bin_ids_val = [
        min(int(all_pos_fracs[i] * n_bins), n_bins - 1) for i in val_idx_list
    ]
    pos_group = _eval_per_group(pos_bin_ids_val)
    pos_results = [
        {**pos_group.get(b, {**nan_result}), "bin": b} for b in range(n_bins)
    ]

    # Axis 2: tool-call index buckets
    tc_bin_ids_val = [
        min(int(all_turn_fracs[i] * n_bins), n_bins - 1) for i in val_idx_list
    ]
    tc_group = _eval_per_group(tc_bin_ids_val)
    tc_results = [
        {**tc_group.get(b, {**nan_result}), "bin": b} for b in range(n_bins)
    ]

    # Axis 3: stages
    stage_ids_val = [all_stages_list[i] for i in val_idx_list]
    stage_results = {
        stage: {**data, "stage": stage}
        for stage, data in _eval_per_group(stage_ids_val).items()
    }

    if save_path is not None:
        _save_probe(probe, mu, std, save_path)

    return pos_results, tc_results, stage_results


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

_STAGE_ORDER = ["exploration", "editing", "testing", "other-bash", "submission", "unknown"]


def _plot_comparison_axis(
    x_labels: list[str],
    per_bucket_accs: list[float],
    single_probe_accs: list[float],
    majority_accs: list[float],
    n_samples: list[int],
    title: str,
    xlabel: str,
    out_path: Path,
) -> None:
    """Plot per-bucket probe accuracy vs single-probe-evaluated-per-bucket."""
    fig, ax = plt.subplots(figsize=(max(8, len(x_labels) * 0.7), 5))
    x = np.arange(len(x_labels))

    ax.plot(x, per_bucket_accs, "o-", color="steelblue", label="Per-bucket probe", linewidth=2)
    ax.plot(x, single_probe_accs, "s--", color="darkorange", label="Single probe (eval per bucket)", linewidth=2)
    ax.plot(x, majority_accs, "^:", color="gray", label="Majority baseline", linewidth=1.5, alpha=0.7)

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
    probes_dir = out_dir / "probes"
    probes_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # W&B init
    # ------------------------------------------------------------------
    wb = None
    if _WANDB_AVAILABLE and not args.wandb_disabled:
        run_name = args.wandb_run_name or f"probes-layer{args.layer}-{out_dir.name}"
        wb = _wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "extract_dir": args.extract_dir,
                "layer": args.layer,
                "n_bins": args.n_bins,
                "max_captures_per_file": args.max_captures_per_file,
                "max_samples_per_bucket": args.max_samples_per_bucket,
                "val_fraction": args.val_fraction,
                "lr_max_iter": args.lr_max_iter,
                "lr_C": args.lr_C,
                "seed": args.seed,
            },
        )
        logger.info(f"W&B run: {wb.url}")

    records = load_records(
        extract_dir, args.layer,
        max_captures_per_file=args.max_captures_per_file,
        seed=args.seed,
    )
    if not records:
        logger.error("No good records found — aborting.")
        if wb:
            wb.finish(exit_code=1)
        return

    n_pass = sum(r.outcome for r in records)
    logger.info(
        f"Records: {len(records)} total, {n_pass} pass / {len(records) - n_pass} fail "
        f"(layer {args.layer})"
    )
    if wb:
        wb.summary["n_records"] = len(records)
        wb.summary["n_pass"] = n_pass
        wb.summary["n_fail"] = len(records) - n_pass

    results: dict = {"layer": args.layer, "n_records": len(records), "n_pass": n_pass}

    # ------------------------------------------------------------------
    # Axis 1: token position bins
    # ------------------------------------------------------------------
    logger.info("=== Axis 1: token position bins ===")
    pos_buckets = bin_by_token_position(records, layer=args.layer, n_bins=args.n_bins)
    pos_results: list[dict] = []
    (probes_dir / "token_position").mkdir(exist_ok=True)
    for bin_idx, (acts, labels) in enumerate(pos_buckets):
        r = _train_probe(acts, labels, args,
                         save_path=probes_dir / "token_position" / f"bin_{bin_idx:02d}.pt")
        r["bin"] = bin_idx
        pos_results.append(r)
        logger.info(
            f"  Bin {bin_idx:2d}: n={r['n']:6d}, "
            f"majority={r['majority']:.3f}, val_acc={r['val_acc']:.3f}"
            + (" [SKIPPED]" if r.get("skipped") else "")
        )
        if wb and not r.get("skipped"):
            wb.log({
                f"token_position/bin_{bin_idx:02d}/val_acc": r["val_acc"],
                f"token_position/bin_{bin_idx:02d}/majority": r["majority"],
                f"token_position/bin_{bin_idx:02d}/n": r["n"],
            })

    fig_pos = out_dir / "probe_token_position.png"
    _plot_axis(
        x_labels=[str(i) for i in range(args.n_bins)],
        val_accs=[r["val_acc"] for r in pos_results],
        majority_accs=[r["majority"] for r in pos_results],
        n_samples=[r["n"] for r in pos_results],
        title=f"Probe accuracy vs token position bin (layer {args.layer})",
        xlabel="Token position bin (0=start, 9=end)",
        out_path=fig_pos,
    )
    if wb:
        wb.log({"figures/token_position": _wandb.Image(str(fig_pos))})
    results["token_position"] = pos_results

    # ------------------------------------------------------------------
    # Axis 2: tool-call index bins
    # ------------------------------------------------------------------
    logger.info("=== Axis 2: tool-call index bins ===")
    tc_buckets = bin_by_tool_call_index(records, layer=args.layer, n_buckets=args.n_bins)
    tc_results: list[dict] = []
    (probes_dir / "tool_call_index").mkdir(exist_ok=True)
    for bin_idx, (acts, labels) in enumerate(tc_buckets):
        r = _train_probe(acts, labels, args,
                         save_path=probes_dir / "tool_call_index" / f"bucket_{bin_idx:02d}.pt")
        r["bin"] = bin_idx
        tc_results.append(r)
        logger.info(
            f"  Bucket {bin_idx:2d}: n={r['n']:6d}, "
            f"majority={r['majority']:.3f}, val_acc={r['val_acc']:.3f}"
            + (" [SKIPPED]" if r.get("skipped") else "")
        )
        if wb and not r.get("skipped"):
            wb.log({
                f"tool_call_index/bucket_{bin_idx:02d}/val_acc": r["val_acc"],
                f"tool_call_index/bucket_{bin_idx:02d}/majority": r["majority"],
                f"tool_call_index/bucket_{bin_idx:02d}/n": r["n"],
            })

    fig_tc = out_dir / "probe_tool_call_index.png"
    _plot_axis(
        x_labels=[str(i) for i in range(args.n_bins)],
        val_accs=[r["val_acc"] for r in tc_results],
        majority_accs=[r["majority"] for r in tc_results],
        n_samples=[r["n"] for r in tc_results],
        title=f"Probe accuracy vs tool-call index bucket (layer {args.layer})",
        xlabel="Tool-call index bucket (0=first turn, 9=last turn)",
        out_path=fig_tc,
    )
    if wb:
        wb.log({"figures/tool_call_index": _wandb.Image(str(fig_tc))})
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
    (probes_dir / "stage").mkdir(exist_ok=True)
    for stage in ordered_stages:
        acts, labels = stage_pools[stage]
        r = _train_probe(acts, labels, args,
                         save_path=probes_dir / "stage" / f"{stage}.pt")
        r["stage"] = stage
        stage_results[stage] = r
        logger.info(
            f"  {stage:15s}: n={r['n']:6d}, "
            f"majority={r['majority']:.3f}, val_acc={r['val_acc']:.3f}"
            + (" [SKIPPED]" if r.get("skipped") else "")
        )
        if wb and not r.get("skipped"):
            wb.log({
                f"stage/{stage}/val_acc": r["val_acc"],
                f"stage/{stage}/majority": r["majority"],
                f"stage/{stage}/n": r["n"],
            })

    fig_stage = out_dir / "probe_stage.png"
    _plot_axis(
        x_labels=ordered_stages,
        val_accs=[stage_results[s]["val_acc"] for s in ordered_stages],
        majority_accs=[stage_results[s]["majority"] for s in ordered_stages],
        n_samples=[stage_results[s]["n"] for s in ordered_stages],
        title=f"Probe accuracy vs tool-call stage (layer {args.layer})",
        xlabel="Stage",
        out_path=fig_stage,
    )
    if wb:
        wb.log({"figures/stage": _wandb.Image(str(fig_stage))})
    results["stage"] = stage_results

    # ------------------------------------------------------------------
    # Global probe: train once on all activations, evaluate per axis
    # ------------------------------------------------------------------
    if args.run_global_probe:
        logger.info("=== Global probe: train once, evaluate per axis ===")
        gp_pos, gp_tc, gp_stage = _train_global_probe_all_axes(
            records, layer=args.layer, n_bins=args.n_bins, args=args,
            save_path=probes_dir / "global_probe.pt",
        )

        for r in gp_pos:
            b = r["bin"]
            if wb and not r.get("skipped"):
                wb.log({
                    f"global_probe/token_position/bin_{b:02d}/val_acc": r["val_acc"],
                    f"global_probe/token_position/bin_{b:02d}/n": r["n"],
                })
        for r in gp_tc:
            b = r["bin"]
            if wb and not r.get("skipped"):
                wb.log({
                    f"global_probe/tool_call_index/bucket_{b:02d}/val_acc": r["val_acc"],
                    f"global_probe/tool_call_index/bucket_{b:02d}/n": r["n"],
                })
        for stage, r in gp_stage.items():
            if wb and not r.get("skipped"):
                wb.log({
                    f"global_probe/stage/{stage}/val_acc": r["val_acc"],
                    f"global_probe/stage/{stage}/n": r["n"],
                })

        fig_pos_cmp = out_dir / "probe_token_position_comparison.png"
        _plot_comparison_axis(
            x_labels=[str(i) for i in range(args.n_bins)],
            per_bucket_accs=[r["val_acc"] for r in pos_results],
            single_probe_accs=[r["val_acc"] for r in gp_pos],
            majority_accs=[r["majority"] for r in pos_results],
            n_samples=[r["n"] for r in pos_results],
            title=f"Per-bin vs global probe: token position (layer {args.layer})",
            xlabel="Token position bin (0=start, 9=end)",
            out_path=fig_pos_cmp,
        )
        if wb:
            wb.log({"figures/token_position_comparison": _wandb.Image(str(fig_pos_cmp))})

        fig_tc_cmp = out_dir / "probe_tool_call_index_comparison.png"
        _plot_comparison_axis(
            x_labels=[str(i) for i in range(args.n_bins)],
            per_bucket_accs=[r["val_acc"] for r in tc_results],
            single_probe_accs=[r["val_acc"] for r in gp_tc],
            majority_accs=[r["majority"] for r in tc_results],
            n_samples=[r["n"] for r in tc_results],
            title=f"Per-bucket vs global probe: tool-call index (layer {args.layer})",
            xlabel="Tool-call index bucket (0=first turn, 9=last turn)",
            out_path=fig_tc_cmp,
        )
        if wb:
            wb.log({"figures/tool_call_index_comparison": _wandb.Image(str(fig_tc_cmp))})

        gp_stage_ordered = [gp_stage.get(s, {"val_acc": float("nan"), "n": 0, "majority": float("nan")})
                            for s in ordered_stages]
        fig_stage_cmp = out_dir / "probe_stage_comparison.png"
        _plot_comparison_axis(
            x_labels=ordered_stages,
            per_bucket_accs=[stage_results[s]["val_acc"] for s in ordered_stages],
            single_probe_accs=[r["val_acc"] for r in gp_stage_ordered],
            majority_accs=[stage_results[s]["majority"] for s in ordered_stages],
            n_samples=[stage_results[s]["n"] for s in ordered_stages],
            title=f"Per-stage vs global probe: stage (layer {args.layer})",
            xlabel="Stage",
            out_path=fig_stage_cmp,
        )
        if wb:
            wb.log({"figures/stage_comparison": _wandb.Image(str(fig_stage_cmp))})

        results["global_probe"] = {
            "token_position": gp_pos,
            "tool_call_index": gp_tc,
            "stage": gp_stage,
        }

    # ------------------------------------------------------------------
    # Save JSON summary
    # ------------------------------------------------------------------
    summary_path = out_dir / "probe_results.json"
    with summary_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved results: {summary_path}")

    if wb:
        wb.save(str(summary_path))
        wb.finish()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = load_from_cli(ProbeRunArgs)
    run_probes(args)


if __name__ == "__main__":
    main()
