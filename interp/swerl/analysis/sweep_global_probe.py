"""W&B hyperparameter sweep for the global outcome probe on SWEbench activations.

Sweeps over architecture (linear / mlp1 / mlp2), learning rate, weight decay,
training epochs, hidden dim, and batch size using Bayesian optimisation.

Each run trains one global probe on all captured activations, logs per-epoch
training curves, and records per-axis bucket accuracy in the run summary for
comparison in the W&B sweep parallel-coordinate and scatter plots.

Memory strategy: records are pre-stacked into a single [N, dim] float32
tensor in main() before any agents run.  The per-record list is freed
immediately after stacking so only one copy lives in RAM during the sweep.

Usage:

    # Create a new sweep and run N agents in this process:
    python -m interp.swerl.analysis.sweep_global_probe \\
        extract_dir=interp-swerl-extract layer=32 n_runs=50

    # Join an existing sweep (to parallelise across machines):
    python -m interp.swerl.analysis.sweep_global_probe \\
        extract_dir=interp-swerl-extract sweep_id=<id> n_runs=20
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import wandb

from cwm.common.params import load_from_cli
from interp.probes.train_probe import build_probe
from interp.swerl.analysis.run_probes import _collect_records, load_records

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sweep search space
# ---------------------------------------------------------------------------

SWEEP_CONFIG: dict = {
    "method": "bayes",
    "metric": {"name": "val_acc", "goal": "maximize"},
    "parameters": {
        "architecture": {"values": ["linear", "mlp1", "mlp2"]},
        "lr": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-2},
        "weight_decay": {"distribution": "log_uniform_values", "min": 0.05, "max": 10.0},
        "n_epochs": {"distribution": "int_uniform", "min": 50, "max": 400},
        "hidden_dim": {"values": [128, 256, 512, 1024]},
        "batch_size": {"values": [256, 512, 1024]},
    },
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SweepArgs:
    extract_dir: str = "interp-swerl-extract"
    layer: int = 32
    n_bins: int = 10
    max_global_probe_samples: int = 200_000
    max_captures_per_file: int = 2_000   # lower than run_probes default to save RAM
    val_fraction: float = 0.2
    seed: int = 42
    n_runs: int = 50
    sweep_id: str = ""          # if set, join this sweep instead of creating one
    wandb_project: str = "cwm-interp"
    wandb_entity: str = ""


# ---------------------------------------------------------------------------
# Agent training function
# ---------------------------------------------------------------------------


def _run_agent(
    *,
    X_all: torch.Tensor,           # [N, dim] float32, pre-stacked
    all_labels: list[int],
    all_pos_fracs: list[float],
    all_turn_fracs: list[float],
    all_stages: list[str],
    n_bins: int,
    max_samples: int,
    val_fraction: float,
    seed: int,
) -> None:
    """One W&B sweep trial.  Called once per hyperparameter sample by wandb.agent."""
    with wandb.init() as run:
        cfg = run.config
        arch: str = cfg.get("architecture", "linear")
        lr: float = cfg.get("lr", 1e-2)
        weight_decay: float = cfg.get("weight_decay", 1.0)
        n_epochs: int = cfg.get("n_epochs", 100)
        hidden_dim: int = cfg.get("hidden_dim", 256)
        batch_size: int = cfg.get("batch_size", 512)

        n = X_all.shape[0]
        if n == 0:
            logger.error("No activations — aborting run.")
            return

        # Subsample indices (operate on indices, not the full tensor, to save RAM)
        if max_samples > 0 and n > max_samples:
            g = torch.Generator().manual_seed(seed)
            keep = torch.randperm(n, generator=g)[:max_samples].sort().values
        else:
            keep = torch.arange(n)

        keep_list = keep.tolist()
        X = X_all[keep]
        y = torch.tensor([all_labels[i] for i in keep_list], dtype=torch.long)
        pos_fracs = [all_pos_fracs[i] for i in keep_list]
        turn_fracs = [all_turn_fracs[i] for i in keep_list]
        stages = [all_stages[i] for i in keep_list]
        n = X.shape[0]

        # Train/val split then standardise (fit on train only)
        n_val = max(1, int(n * val_fraction))
        n_train = n - n_val
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=g)
        train_idx, val_idx = perm[:n_train], perm[n_train:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        mu = X_train.mean(0, keepdim=True)
        std = X_train.std(0, keepdim=True).clamp(min=1e-6)
        X_train = (X_train - mu) / std
        X_val = (X_val - mu) / std

        dim = X_train.shape[1]
        probe = build_probe(arch, dim, 2, hidden_dim)
        optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        train_loader = DataLoader(
            TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
        )

        best_val_acc = 0.0
        for epoch in range(n_epochs):
            probe.train()
            total_loss = total_correct = total_n = 0
            for xb, yb in train_loader:
                logits = probe(xb)
                loss = criterion(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(probe.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item() * len(yb)
                total_correct += (logits.argmax(-1) == yb).sum().item()
                total_n += len(yb)

            probe.eval()
            with torch.no_grad():
                val_acc = (probe(X_val).argmax(-1) == y_val).float().mean().item()

            best_val_acc = max(best_val_acc, val_acc)
            run.log({
                "train_loss": total_loss / total_n if total_n > 0 else 0.0,
                "train_acc": total_correct / total_n if total_n > 0 else 0.0,
                "val_acc": val_acc,
                "epoch": epoch,
            })

        run.summary["val_acc"] = best_val_acc

        # Per-axis bucket accuracy on the val split
        probe.eval()
        val_idx_list = val_idx.tolist()
        with torch.no_grad():
            val_preds = probe(X_val).argmax(-1)

        # Axis 1: token position bins
        pos_bins_val = [min(int(pos_fracs[i] * n_bins), n_bins - 1) for i in val_idx_list]
        for b in range(n_bins):
            mask = torch.tensor([g == b for g in pos_bins_val], dtype=torch.bool)
            if mask.any():
                acc = (val_preds[mask] == y_val[mask]).float().mean().item()
                run.summary[f"token_position/bin_{b:02d}"] = acc

        # Axis 2: tool-call index bins
        tc_bins_val = [min(int(turn_fracs[i] * n_bins), n_bins - 1) for i in val_idx_list]
        for b in range(n_bins):
            mask = torch.tensor([g == b for g in tc_bins_val], dtype=torch.bool)
            if mask.any():
                acc = (val_preds[mask] == y_val[mask]).float().mean().item()
                run.summary[f"tool_call_index/bin_{b:02d}"] = acc

        # Axis 3: stage
        stages_val = [stages[i] for i in val_idx_list]
        for stage in sorted(set(stages_val)):
            mask = torch.tensor([s == stage for s in stages_val], dtype=torch.bool)
            if mask.any():
                acc = (val_preds[mask] == y_val[mask]).float().mean().item()
                run.summary[f"stage/{stage}"] = acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = load_from_cli(SweepArgs)

    entity = args.wandb_entity or None

    logger.info(f"Loading records from {args.extract_dir}, layer={args.layer} ...")
    records = load_records(
        Path(args.extract_dir),
        layer=args.layer,
        max_captures_per_file=args.max_captures_per_file,
        seed=args.seed,
    )
    logger.info(f"Loaded {len(records)} records")

    all_acts, all_labels, all_pos_fracs, all_turn_fracs, all_stages = _collect_records(
        records, args.layer
    )
    n_total = len(all_acts)
    logger.info(f"Total tokens: {n_total} — stacking into fp16 tensor ...")

    # Pre-stack once, then free per-row list and original records to save RAM.
    # Peak during stack: ~2× the tensor size; after del: only X_all remains.
    X_all = torch.stack(all_acts)   # [N, dim] float32
    del all_acts, records
    logger.info(
        f"X_all shape: {list(X_all.shape)}, "
        f"RAM ~{X_all.numel() * 4 / 1e9:.1f} GB (float32)"
    )

    agent_fn = partial(
        _run_agent,
        X_all=X_all,
        all_labels=all_labels,
        all_pos_fracs=all_pos_fracs,
        all_turn_fracs=all_turn_fracs,
        all_stages=all_stages,
        n_bins=args.n_bins,
        max_samples=args.max_global_probe_samples,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    if args.sweep_id:
        sweep_id = args.sweep_id
        logger.info(f"Joining existing sweep: {sweep_id}")
    else:
        sweep_id = wandb.sweep(
            SWEEP_CONFIG,
            project=args.wandb_project,
            entity=entity,
        )
        logger.info(f"Created sweep: {sweep_id}")
        logger.info(
            f"View at: https://wandb.ai/"
            f"{entity or '<entity>'}/{args.wandb_project}/sweeps/{sweep_id}"
        )

    wandb.agent(
        sweep_id,
        function=agent_fn,
        count=args.n_runs,
        project=args.wandb_project,
        entity=entity,
    )


if __name__ == "__main__":
    main()
