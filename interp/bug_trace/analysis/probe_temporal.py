"""Train linear probes on decode-step trajectories and produce T×L accuracy heatmaps.

For each (layer L, relative time bin B), extracts h_T for steps T in bin B,
trains a linear probe to predict `is_buggy` / `will_be_correct`, and records accuracy.

Three-way split (all at original_id level when split_by=original):
  - train       : fit probe weights
  - val         : early stopping only
  - test        : reported only in single-run mode, never used during sweep

Single run:
    python -m interp.bug_trace.analysis.probe_temporal \\
        traj_dir=./interp-bug-trajectories-track_a \\
        target=will_be_correct split_by=original

Hyperparameter sweep (create new):
    python -m interp.bug_trace.analysis.probe_temporal \\
        traj_dir=./interp-bug-trajectories-track_a \\
        target=will_be_correct split_by=original n_runs=30

Join existing sweep:
    python -m interp.bug_trace.analysis.probe_temporal \\
        traj_dir=./interp-bug-trajectories-track_a \\
        target=will_be_correct split_by=original n_runs=20 sweep_id=<id>
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, TensorDataset

from cwm.common.params import load_from_cli

logger = logging.getLogger(__name__)


SWEEP_CONFIG: dict = {
    "method": "bayes",
    "metric": {"name": "val_acc_L32_mean", "goal": "maximize"},
    "parameters": {
        "lr":           {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-2},
        "weight_decay": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-1},
        "batch_size":   {"values": [256, 512, 1024]},
        "max_epochs":   {"values": [50, 100, 200, 500]},
        "patience":     {"values": [5, 10, 20]},
    },
}


@dataclass
class ProbeTemporalArgs:
    traj_dir: str = "interp-bug-trajectories"
    target: str = "is_buggy"           # is_buggy | will_be_correct
    layers: list[int] = field(default_factory=lambda: [16, 32, 48, 63])
    n_time_bins: int = 10
    # Single-run hyperparams (ignored during sweep)
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    patience: int = 5
    num_workers: int = 4
    # Split — all boundaries applied at original_id level when split_by=original
    val_fraction: float = 0.15
    test_fraction: float = 0.15        # held out; never seen during sweep
    split_by: str = "sample"           # "sample" | "original"
    seed: int = 42
    # W&B / sweep
    wandb_project: str = "cwm-interp"
    wandb_run_name: str = ""
    n_runs: int = 0                    # 0 = single run; >0 = sweep
    sweep_id: str = ""                 # empty = create new sweep; non-empty = join existing


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _compute_splits(
    index: list[dict],
    split_by: str,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    """Return (val_ids, test_ids) at the sample_id level.

    split_by='original': boundaries are drawn at original_id level so all
    buggy variants of the same program land on the same side.
    """
    rng = random.Random(seed)

    if split_by == "original":
        original_ids = sorted({m["original_id"] for m in index})
        n = len(original_ids)
        n_val  = max(1, int(n * val_fraction))
        n_test = max(1, int(n * test_fraction))
        shuffled = rng.sample(original_ids, n)
        val_originals  = set(shuffled[:n_val])
        test_originals = set(shuffled[n_val:n_val + n_test])
        val_ids  = {m["sample_id"] for m in index if m["original_id"] in val_originals}
        test_ids = {m["sample_id"] for m in index if m["original_id"] in test_originals}
        n_train_orig = n - n_val - n_test
        logger.info(
            f"Split by original_id: {n_train_orig} train / {n_val} val / {n_test} test originals "
            f"→ {len(index)-len(val_ids)-len(test_ids)} train / {len(val_ids)} val / {len(test_ids)} test trajectories"
        )
    else:
        all_sids = [m["sample_id"] for m in index]
        n = len(all_sids)
        n_val  = max(1, int(n * val_fraction))
        n_test = max(1, int(n * test_fraction))
        shuffled = rng.sample(all_sids, n)
        val_ids  = set(shuffled[:n_val])
        test_ids = set(shuffled[n_val:n_val + n_test])
        logger.info(
            f"Split by sample_id: {n-n_val-n_test} train / {n_val} val / {n_test} test trajectories"
        )

    return val_ids, test_ids


def _load_data(
    index: list[dict],
    val_ids: set[str],
    test_ids: set[str],
    traj_subdir: Path,
    layers: list[int],
    label_fn,
) -> dict[int, tuple]:
    """Load all trajectories and return per-layer stacked tensors.

    Returns {layer: (H_all, y_all, rp_all, val_mask, test_mask)}.
    train = ~val_mask & ~test_mask.
    """
    layer_h:       dict[int, list] = {L: [] for L in layers}
    layer_labels:  dict[int, list] = {L: [] for L in layers}
    layer_relpos:  dict[int, list] = {L: [] for L in layers}
    layer_is_val:  dict[int, list] = {L: [] for L in layers}
    layer_is_test: dict[int, list] = {L: [] for L in layers}

    for meta in index:
        sid = meta["sample_id"]
        pt = traj_subdir / f"{sid}.pt"
        if not pt.exists():
            continue
        sample = torch.load(pt, map_location="cpu", weights_only=False)
        traj = sample.get("trajectory", {})
        label   = label_fn(meta)
        is_val  = sid in val_ids
        is_test = sid in test_ids

        for layer in layers:
            h_traj = traj.get(layer)
            if h_traj is None:
                h_traj = traj.get(str(layer))
            if h_traj is None:
                continue
            T = h_traj.shape[0]
            for t in range(T):
                layer_h[layer].append(h_traj[t])
                layer_labels[layer].append(label)
                layer_relpos[layer].append(t / max(T - 1, 1))
                layer_is_val[layer].append(is_val)
                layer_is_test[layer].append(is_test)

    data = {}
    for layer in layers:
        if not layer_h[layer]:
            continue
        data[layer] = (
            torch.stack(layer_h[layer]),
            torch.tensor(layer_labels[layer], dtype=torch.long),
            torch.tensor(layer_relpos[layer]),
            torch.tensor(layer_is_val[layer]),
            torch.tensor(layer_is_test[layer]),
        )
    return data


# ---------------------------------------------------------------------------
# Probe training
# ---------------------------------------------------------------------------

def _train_linear_probe(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    n_classes: int,
    max_epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    patience: int,
    num_workers: int = 0,
    log_prefix: str = "",
    X_test: torch.Tensor | None = None,
    y_test: torch.Tensor | None = None,
) -> dict:
    """Train a linear probe with early stopping.

    Early stopping uses val_acc. Test is evaluated once at the end using the
    best weights — only pass X_test/y_test for the definitive single run.
    """
    if X_train.shape[0] < 10 or X_val.shape[0] < 1:
        return {"val_acc": float("nan"), "n_train": X_train.shape[0], "n_val": X_val.shape[0]}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train = X_train.float().to(device)
    y_train = y_train.to(device)
    X_val   = X_val.float().to(device)
    y_val   = y_val.to(device)

    # Mean-center using train statistics only
    mu      = X_train.mean(0, keepdim=True)
    X_train = X_train - mu
    X_val   = X_val - mu

    probe = nn.Linear(X_train.shape[1], n_classes).to(device)
    opt   = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    crit  = nn.CrossEntropyLoss()

    loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )

    best_val_acc   = -1.0
    best_weights   = None
    patience_count = 0

    for epoch in range(max_epochs):
        probe.train()
        epoch_loss = 0.0
        n_batches  = 0
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(probe(xb), yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches  += 1

        probe.eval()
        with torch.no_grad():
            val_acc = (probe(X_val).argmax(1) == y_val).float().mean().item()

        if log_prefix:
            wandb.log({
                f"{log_prefix}/train_loss": epoch_loss / max(n_batches, 1),
                f"{log_prefix}/val_acc":    val_acc,
                f"{log_prefix}/epoch":      epoch,
            })

        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            best_weights   = (probe.weight.detach().clone(), probe.bias.detach().clone())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break

    best_w, best_b = best_weights
    result = {
        "val_acc":       best_val_acc,
        "n_train":       X_train.shape[0],
        "n_val":         X_val.shape[0],
        "stopped_epoch": epoch,
        "probe_weight":  best_w.cpu(),
        "probe_bias":    best_b.cpu(),
        "feature_mean":  mu.cpu(),
    }

    # Evaluate on test set using best weights (single-run mode only)
    if X_test is not None and y_test is not None and X_test.shape[0] > 0:
        X_test = (X_test.float().to(device)) - mu
        probe.weight.data = best_w.to(device)
        probe.bias.data   = best_b.to(device)
        probe.eval()
        with torch.no_grad():
            test_acc = (probe(X_test).argmax(1) == y_test.to(device)).float().mean().item()
        result["test_acc"] = test_acc
        result["n_test"]   = X_test.shape[0]

    return result


# ---------------------------------------------------------------------------
# Probe loop
# ---------------------------------------------------------------------------

def _run_probes(
    data: dict,
    layers: list[int],
    n_time_bins: int,
    n_classes: int,
    max_epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    patience: int,
    num_workers: int = 0,
    log_curves: bool = True,
    include_test: bool = False,
) -> dict:
    """Train probes for all (layer, bin). include_test=True only for final single run."""
    heatmap: dict = {}
    for layer in layers:
        if layer not in data:
            continue
        H_all, y_all, rp_all, val_mask_all, test_mask_all = data[layer]
        train_mask_all = ~val_mask_all & ~test_mask_all
        heatmap[layer] = {}

        for bin_idx in range(n_time_bins):
            lo, hi   = bin_idx / n_time_bins, (bin_idx + 1) / n_time_bins
            bin_mask = (rp_all >= lo) & (rp_all < hi)
            if not bin_mask.any():
                continue

            X_train = H_all[bin_mask & train_mask_all].float()
            y_train = y_all[bin_mask & train_mask_all]
            X_val   = H_all[bin_mask & val_mask_all].float()
            y_val   = y_all[bin_mask & val_mask_all]
            X_test  = H_all[bin_mask & test_mask_all].float() if include_test else None
            y_test  = y_all[bin_mask & test_mask_all]         if include_test else None

            result = _train_linear_probe(
                X_train, y_train, X_val, y_val,
                n_classes, max_epochs, lr, weight_decay, batch_size, patience,
                num_workers=num_workers,
                log_prefix=f"L{layer}/bin{bin_idx:02d}" if log_curves else "",
                X_test=X_test, y_test=y_test,
            )
            heatmap[layer][bin_idx] = result

            test_str = f" test_acc={result['test_acc']:.3f} (n_test={result['n_test']})" \
                       if "test_acc" in result else ""
            logger.info(
                f"L={layer} bin={bin_idx}/{n_time_bins} "
                f"val_acc={result['val_acc']:.3f} stopped_epoch={result.get('stopped_epoch','?')} "
                f"(n_train={result['n_train']} n_val={result['n_val']}){test_str}"
            )
    return heatmap


def _print_heatmap(heatmap: dict, layers: list[int], n_time_bins: int, target: str) -> None:
    for metric, label in [("val_acc", "val"), ("test_acc", "test")]:
        # Only print test if at least one cell has it
        has_metric = any(
            metric in heatmap.get(l, {}).get(b, {})
            for l in layers for b in range(n_time_bins)
        )
        if not has_metric:
            continue
        print(f"\n=== Probe Temporal Heatmap ({label}): {target} ===")
        print(f"{'Layer':>6}", end="")
        for b in range(n_time_bins):
            print(f"  bin{b:02d}", end="")
        print()
        for layer in layers:
            print(f"{layer:>6}", end="")
            for b in range(n_time_bins):
                v = heatmap.get(layer, {}).get(b, {}).get(metric, float("nan"))
                print(f"  {'nan':>5}" if v != v else f"  {v:.3f}", end="")
            print()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_probe_temporal(args: ProbeTemporalArgs) -> None:
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(args.seed)

    traj_path = Path(args.traj_dir)

    index: list[dict] = []
    with (traj_path / "index.jsonl").open() as f:
        for line in f:
            meta = json.loads(line)
            if meta.get("is_buggy", False):
                index.append(meta)

    val_ids, test_ids = _compute_splits(
        index, args.split_by, args.val_fraction, args.test_fraction, args.seed
    )

    label_fn = {
        "is_buggy":        lambda m: int(m.get("is_buggy", False)),
        "will_be_correct": lambda m: int(m.get("correct", False)),
    }.get(args.target)
    if label_fn is None:
        raise ValueError(f"Unknown target: {args.target!r}")

    logger.info("Loading trajectory data...")
    data = _load_data(
        index, val_ids, test_ids,
        traj_path / "trajectories", args.layers, label_fn,
    )
    logger.info("Data loaded.")

    n_classes = 2

    if args.n_runs == 0:
        # ----- single run — test set is evaluated -----
        run_name = args.wandb_run_name or f"probe_{args.target}_{args.split_by}"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
        heatmap = _run_probes(
            data, args.layers, args.n_time_bins, n_classes,
            max_epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
            batch_size=args.batch_size, patience=args.patience,
            num_workers=args.num_workers, include_test=True,
        )
        columns_val  = ["layer"] + [f"bin{b:02d}" for b in range(args.n_time_bins)]
        columns_test = ["layer"] + [f"bin{b:02d}" for b in range(args.n_time_bins)]
        val_rows, test_rows = [], []
        for layer in args.layers:
            val_rows.append(
                [layer] + [heatmap.get(layer, {}).get(b, {}).get("val_acc",  float("nan"))
                           for b in range(args.n_time_bins)]
            )
            test_rows.append(
                [layer] + [heatmap.get(layer, {}).get(b, {}).get("test_acc", float("nan"))
                           for b in range(args.n_time_bins)]
            )
            for bin_idx in range(args.n_time_bins):
                cell = heatmap.get(layer, {}).get(bin_idx, {})
                wandb.summary[f"L{layer}/bin{bin_idx:02d}/val_acc"]  = cell.get("val_acc",  float("nan"))
                wandb.summary[f"L{layer}/bin{bin_idx:02d}/test_acc"] = cell.get("test_acc", float("nan"))

        wandb.log({
            "heatmap_val":  wandb.Table(columns=columns_val,  data=val_rows),
            "heatmap_test": wandb.Table(columns=columns_test, data=test_rows),
        })
        wandb.finish()

        out = traj_path / f"probe_temporal_{args.target}.pt"
        torch.save({"heatmap": heatmap, "args": vars(args)}, out)
        logger.info(f"Saved to {out}")
        _print_heatmap(heatmap, args.layers, args.n_time_bins, args.target)

    else:
        # ----- sweep — test set is never touched -----
        def _agent_fn():
            with wandb.init():
                cfg = wandb.config
                heatmap = _run_probes(
                    data, args.layers, args.n_time_bins, n_classes,
                    max_epochs=cfg.max_epochs, lr=cfg.lr,
                    weight_decay=cfg.weight_decay, batch_size=cfg.batch_size,
                    patience=cfg.patience, num_workers=args.num_workers,
                    include_test=False,
                )
                l32_accs = [
                    heatmap.get(32, {}).get(b, {}).get("val_acc", float("nan"))
                    for b in range(args.n_time_bins)
                ]
                valid = [v for v in l32_accs if v == v]
                wandb.log({"val_acc_L32_mean": sum(valid) / len(valid) if valid else float("nan")})
                for layer in args.layers:
                    for bin_idx in range(args.n_time_bins):
                        v = heatmap.get(layer, {}).get(bin_idx, {}).get("val_acc", float("nan"))
                        wandb.summary[f"L{layer}/bin{bin_idx:02d}/val_acc"] = v

        sweep_config = {**SWEEP_CONFIG, "name": f"probe_{args.target}_{args.split_by}"}
        if args.sweep_id:
            sweep_id = args.sweep_id
            logger.info(f"Joining sweep {sweep_id}")
        else:
            sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)
            logger.info(f"Created sweep {sweep_id}")

        wandb.agent(sweep_id, function=_agent_fn, count=args.n_runs, project=args.wandb_project)


if __name__ == "__main__":
    args = load_from_cli(ProbeTemporalArgs)
    run_probe_temporal(args)
