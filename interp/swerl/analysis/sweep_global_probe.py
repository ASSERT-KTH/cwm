"""W&B hyperparameter sweep for the global outcome probe on SWEbench activations.

Sweeps over architecture (linear / mlp1), learning rate, weight decay,
training epochs, hidden dim, and batch size using Bayesian optimisation.

Each run trains one global probe on all captured activations, logs per-epoch
training curves, and records per-axis bucket accuracy in the run summary for
comparison in the W&B sweep parallel-coordinate and scatter plots.

Memory strategy: records are pre-stacked into a single [N, dim] float32
tensor in main() before any agents run.  The per-record list is freed
immediately after stacking so only one copy lives in RAM during the sweep.
A cache file can be written on the first run and reused by subsequent agents
to skip the 485-file load entirely.

Train/val split is trajectory-level: all tokens from a given SWEbench instance
go entirely to train or entirely to val, preventing within-trajectory leakage.

Usage:

    # Build the cache only (n_runs=0), then exit:
    python -m interp.swerl.analysis.sweep_global_probe \\
        extract_dir=interp-swerl-extract cache_path=probe_cache_layer32.pt n_runs=0

    # Create a new sweep and run N agents in this process:
    python -m interp.swerl.analysis.sweep_global_probe \\
        extract_dir=interp-swerl-extract cache_path=probe_cache_layer32.pt n_runs=50

    # Join an existing sweep (to parallelise across machines):
    python -m interp.swerl.analysis.sweep_global_probe \\
        cache_path=probe_cache_layer32.pt sweep_id=<id> n_runs=20
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
        "architecture": {"values": ["linear", "mlp1"]},
        "lr": {"distribution": "log_uniform_values", "min": 1e-5, "max": 5e-4},
        "weight_decay": {"distribution": "log_uniform_values", "min": 1e-3, "max": 0.5},
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
    cache_path: str = ""        # if set, save/load pre-stacked tensor to skip 485-file reload


# ---------------------------------------------------------------------------
# Agent training function
# ---------------------------------------------------------------------------


def _run_agent(
    *,
    X_all: torch.Tensor,              # [N, dim] float32, CPU
    all_labels: list[int],
    all_pos_fracs: list[float],
    all_turn_fracs: list[float],
    all_stages: list[str],
    all_instance_ids: list[str],
    n_bins: int,
    max_samples: int,
    val_fraction: float,
    seed: int,
) -> None:
    """One W&B sweep trial.  Called once per hyperparameter sample by wandb.agent."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with wandb.init() as run:
        cfg = run.config
        arch: str = cfg.get("architecture", "linear")
        lr: float = cfg.get("lr", 1e-3)
        weight_decay: float = cfg.get("weight_decay", 0.1)
        n_epochs: int = cfg.get("n_epochs", 100)
        hidden_dim: int = cfg.get("hidden_dim", 256)
        batch_size: int = cfg.get("batch_size", 512)

        n = X_all.shape[0]
        if n == 0:
            logger.error("No activations — aborting run.")
            return

        # Trajectory-level train/val split: shuffle unique instance IDs, split 80/20,
        # then assign every token from each instance to the appropriate partition.
        unique_ids = sorted(set(all_instance_ids))
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(unique_ids), generator=g).tolist()
        n_val_traj = max(1, int(len(unique_ids) * val_fraction))
        val_ids = {unique_ids[i] for i in perm[-n_val_traj:]}

        train_indices = [i for i, iid in enumerate(all_instance_ids) if iid not in val_ids]
        val_indices   = [i for i, iid in enumerate(all_instance_ids) if iid in val_ids]

        # Subsample train tokens if needed to cap RAM / compute
        if max_samples > 0 and len(train_indices) > max_samples:
            g2 = torch.Generator().manual_seed(seed)
            perm2 = torch.randperm(len(train_indices), generator=g2)[:max_samples].tolist()
            train_indices = [train_indices[i] for i in sorted(perm2)]

        X_train = X_all[train_indices].to(device)
        y_train = torch.tensor([all_labels[i] for i in train_indices], dtype=torch.long, device=device)
        X_val   = X_all[val_indices].to(device)
        y_val   = torch.tensor([all_labels[i] for i in val_indices], dtype=torch.long, device=device)

        run.log({"n_train": len(train_indices), "n_val": len(val_indices), "n_val_traj": n_val_traj})

        mu = X_train.mean(0, keepdim=True)
        std = X_train.std(0, keepdim=True).clamp(min=1e-6)
        X_train = (X_train - mu) / std
        X_val   = (X_val   - mu) / std

        dim = X_train.shape[1]
        probe = build_probe(arch, dim, 2, hidden_dim).to(device)
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

        # Per-axis bucket accuracy — move predictions to CPU for list-based masking
        probe.eval()
        with torch.no_grad():
            val_preds = probe(X_val).argmax(-1).cpu()
        y_val_cpu = y_val.cpu()

        pos_fracs_val  = [all_pos_fracs[i]  for i in val_indices]
        turn_fracs_val = [all_turn_fracs[i] for i in val_indices]
        stages_val     = [all_stages[i]     for i in val_indices]

        # Axis 1: token position bins
        pos_bins = [min(int(f * n_bins), n_bins - 1) for f in pos_fracs_val]
        for b in range(n_bins):
            mask = torch.tensor([x == b for x in pos_bins], dtype=torch.bool)
            if mask.any():
                run.summary[f"token_position/bin_{b:02d}"] = (val_preds[mask] == y_val_cpu[mask]).float().mean().item()

        # Axis 2: tool-call index bins
        tc_bins = [min(int(f * n_bins), n_bins - 1) for f in turn_fracs_val]
        for b in range(n_bins):
            mask = torch.tensor([x == b for x in tc_bins], dtype=torch.bool)
            if mask.any():
                run.summary[f"tool_call_index/bin_{b:02d}"] = (val_preds[mask] == y_val_cpu[mask]).float().mean().item()

        # Axis 3: stage
        for stage in sorted(set(stages_val)):
            mask = torch.tensor([s == stage for s in stages_val], dtype=torch.bool)
            if mask.any():
                run.summary[f"stage/{stage}"] = (val_preds[mask] == y_val_cpu[mask]).float().mean().item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = load_from_cli(SweepArgs)

    entity = args.wandb_entity or None
    cache = Path(args.cache_path) if args.cache_path else None

    if cache is not None and cache.exists():
        logger.info(f"Loading cache from {cache} ...")
        cached = torch.load(cache, map_location="cpu", weights_only=False)
        X_all = cached["X_all"]
        all_labels = cached["all_labels"]
        all_pos_fracs = cached["all_pos_fracs"]
        all_turn_fracs = cached["all_turn_fracs"]
        all_stages = cached["all_stages"]
        all_instance_ids = cached.get("all_instance_ids", [""] * X_all.shape[0])
        logger.info(f"Loaded cache: X_all shape={list(X_all.shape)}, "
                    f"unique trajectories={len(set(all_instance_ids))}")
    else:
        logger.info(f"Loading records from {args.extract_dir}, layer={args.layer} ...")
        records = load_records(
            Path(args.extract_dir),
            layer=args.layer,
            max_captures_per_file=args.max_captures_per_file,
            seed=args.seed,
        )
        logger.info(f"Loaded {len(records)} records")

        all_acts, all_labels, all_pos_fracs, all_turn_fracs, all_stages, all_instance_ids = (
            _collect_records(records, args.layer)
        )
        n_total = len(all_acts)
        logger.info(f"Total tokens: {n_total} — stacking into float32 tensor ...")

        X_all = torch.stack(all_acts).float()   # [N, dim] float32
        del all_acts, records
        logger.info(
            f"X_all shape: {list(X_all.shape)}, "
            f"RAM ~{X_all.numel() * 4 / 1e9:.1f} GB (float32)"
        )

        if cache is not None:
            logger.info(f"Saving cache to {cache} ...")
            torch.save({
                "X_all": X_all,
                "all_labels": all_labels,
                "all_pos_fracs": all_pos_fracs,
                "all_turn_fracs": all_turn_fracs,
                "all_stages": all_stages,
                "all_instance_ids": all_instance_ids,
            }, cache)
            logger.info(f"Cache saved ({cache.stat().st_size / 1e9:.1f} GB)")

    if args.n_runs == 0:
        logger.info("n_runs=0 — cache built, exiting.")
        return

    agent_fn = partial(
        _run_agent,
        X_all=X_all,
        all_labels=all_labels,
        all_pos_fracs=all_pos_fracs,
        all_turn_fracs=all_turn_fracs,
        all_stages=all_stages,
        all_instance_ids=all_instance_ids,
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
