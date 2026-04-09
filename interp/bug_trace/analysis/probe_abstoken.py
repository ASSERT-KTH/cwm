"""Probe accuracy vs absolute generated token position.

Unlike probe_temporal.py (which uses relative position t/(T-1) → equal-width bins),
this script buckets by absolute token count: step t at stride S → token position t*S.

This reveals how probe accuracy evolves over actual generation length, and whether
accuracy improvements at late stages are driven by the extended 16k context
(long generations unavailable at 4k).

Produces two plots:
  - mutation_type probe (buggy samples only, 3-class hard mutations)
  - will_be_correct probe (buggy samples only, binary)

Usage:
    python -m interp.bug_trace.analysis.probe_abstoken \\
        traj_dir=./interp-bug-trajectories-hard-16k \\
        bucket_width=200 \\
        layers=[16,32,48,63]
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from cwm.common.params import load_from_cli

logger = logging.getLogger(__name__)

HARD_MUTATION_TYPES = [
    "deleted_accumulator",
    "swapped_arguments",
    "wrong_variable",
]

ALL_MUTATION_TYPES = HARD_MUTATION_TYPES + [
    "condition_flip",
    "off_by_one_minus",
    "off_by_one_plus",
    "wrong_comparator",
    "wrong_operator",
]


@dataclass
class ProbeAbsTokenArgs:
    traj_dir: str = "interp-bug-trajectories-hard-16k"
    layers: list[int] = field(default_factory=lambda: [16, 32, 48, 63])
    bucket_width: int = 200     # tokens per bucket
    min_samples: int = 10       # skip bucket if fewer samples than this
    epochs: int = 30
    lr: float = 1e-3
    batch_size: int = 512
    val_fraction: float = 0.2
    seed: int = 42
    targets: list[str] = field(default_factory=lambda: ["mutation_type", "will_be_correct"])


def _train_probe(X: torch.Tensor, y: torch.Tensor, n_classes: int,
                 epochs: int, lr: float, batch_size: int,
                 val_fraction: float, seed: int) -> dict:
    if X.shape[0] < 10:
        return {"val_acc": float("nan"), "n": X.shape[0]}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = (X.float() - X.float().mean(0, keepdim=True)).to(device)
    y = y.to(device)

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
        val_acc = (probe(Xv).argmax(1) == yv).float().mean().item()

    return {"val_acc": val_acc, "n": n}


def _load_index(traj_path: Path) -> list[dict]:
    index = []
    with (traj_path / "index.jsonl").open() as f:
        for line in f:
            index.append(json.loads(line))
    return index


def run_probe_abstoken(args: ProbeAbsTokenArgs) -> None:
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(args.seed)

    traj_path = Path(args.traj_dir)
    traj_subdir = traj_path / "trajectories"
    figures_dir = Path("experiments/02_bug_trace/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    index = _load_index(traj_path)

    # Detect which mutation types are present
    present_types = sorted({
        m["mutation_type"] for m in index
        if m.get("is_buggy") and m.get("mutation_type") in ALL_MUTATION_TYPES
    })
    logger.info(f"Present mutation types: {present_types}")
    local_to_idx = {mt: i for i, mt in enumerate(present_types)}

    # ---------- collect data: one pass over .pt files per layer ----------
    # For each sample we record (step, absolute_token_pos, h_vector, mut_label, wbc_label)
    # We store per-layer to avoid holding all layers in memory simultaneously.

    for layer in args.layers:
        logger.info(f"\n=== Layer {layer} ===")

        # buckets[bucket_idx] = {"mut": [(h, label), ...], "wbc": [(h, label), ...]}
        mut_buckets: dict[int, list] = defaultdict(list)
        wbc_buckets: dict[int, list] = defaultdict(list)

        for meta in index:
            is_buggy = meta.get("is_buggy", False)
            mut_type = meta.get("mutation_type", "")
            correct = meta.get("correct", False)

            sid = meta["sample_id"]
            pt = traj_subdir / f"{sid}.pt"
            if not pt.exists():
                continue
            sample = torch.load(pt, map_location="cpu", weights_only=False)
            traj = sample.get("trajectory", {})

            h_traj = traj.get(layer)
            if h_traj is None:
                h_traj = traj.get(str(layer))
            if h_traj is None:
                continue

            stride = sample.get("stride", 5)
            T = h_traj.shape[0]

            for t in range(T):
                abs_tok = t * stride
                bucket_idx = abs_tok // args.bucket_width

                h = h_traj[t].half()  # keep fp16 to save memory

                # mutation_type: only buggy samples with known type
                if is_buggy and mut_type in local_to_idx:
                    mut_buckets[bucket_idx].append((h, local_to_idx[mut_type]))

                # will_be_correct: only buggy samples
                if is_buggy:
                    wbc_buckets[bucket_idx].append((h, int(correct)))

        # train probes per bucket
        max_bucket = max(
            (max(mut_buckets.keys(), default=0), max(wbc_buckets.keys(), default=0))
        )
        n_buckets = max_bucket + 1

        mut_results = {}   # bucket_idx → {val_acc, n, center_tok}
        wbc_results = {}

        for bucket_idx in range(n_buckets):
            center_tok = bucket_idx * args.bucket_width + args.bucket_width // 2

            # mutation_type
            if "mutation_type" in args.targets and len(mut_buckets[bucket_idx]) >= args.min_samples:
                pairs = mut_buckets[bucket_idx]
                H = torch.stack([p[0] for p in pairs])
                y = torch.tensor([p[1] for p in pairs], dtype=torch.long)
                res = _train_probe(H, y, len(present_types),
                                   args.epochs, args.lr, args.batch_size,
                                   args.val_fraction, args.seed)
                res["center_tok"] = center_tok
                res["n"] = len(pairs)
                mut_results[bucket_idx] = res
                logger.info(f"  [mut] bucket {bucket_idx} ({center_tok} tok): "
                            f"acc={res['val_acc']:.3f} n={len(pairs)}")

            # will_be_correct
            if "will_be_correct" in args.targets and len(wbc_buckets[bucket_idx]) >= args.min_samples:
                pairs = wbc_buckets[bucket_idx]
                H = torch.stack([p[0] for p in pairs])
                y = torch.tensor([p[1] for p in pairs], dtype=torch.long)
                res = _train_probe(H, y, 2,
                                   args.epochs, args.lr, args.batch_size,
                                   args.val_fraction, args.seed)
                res["center_tok"] = center_tok
                res["n"] = len(pairs)
                wbc_results[bucket_idx] = res
                logger.info(f"  [wbc] bucket {bucket_idx} ({center_tok} tok): "
                            f"acc={res['val_acc']:.3f} n={len(pairs)}")

        # save per-layer results
        torch.save(
            {"mut": mut_results, "wbc": wbc_results,
             "layer": layer, "args": vars(args),
             "mutation_types": present_types},
            traj_path / f"probe_abstoken_layer{layer}.pt"
        )

    # ---------- load all layers and plot ----------
    _plot_results(args, traj_path, figures_dir, present_types)


def _plot_results(args: ProbeAbsTokenArgs, traj_path: Path,
                  figures_dir: Path, present_types: list[str]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except Exception as e:
        logger.warning(f"matplotlib unavailable: {e}. Skipping plots.")
        return

    COLORS = {16: "#1f77b4", 32: "#d62728", 48: "#2ca02c", 63: "#9467bd"}

    def _safe_plot(target_key: str, title: str, ylabel: str,
                   baseline: float, baseline_label: str, fname: str) -> None:
        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax2 = ax1.twinx()

        for layer in args.layers:
            pt = traj_path / f"probe_abstoken_layer{layer}.pt"
            if not pt.exists():
                continue
            data = torch.load(pt, map_location="cpu", weights_only=False)
            results = data[target_key]
            if not results:
                continue

            xs = [r["center_tok"] for r in results.values()]
            ys = [r["val_acc"] for r in results.values()]
            ns = [r["n"] for r in results.values()]
            color = COLORS.get(layer, None)

            ax1.plot(xs, ys, color=color, linewidth=2, label=f"L{layer}")
            ax2.fill_between(xs, ns, alpha=0.07, color=color)

        ax1.axhline(baseline, color="gray", linestyle="--", linewidth=1.2,
                    label=baseline_label)
        # Vertical line at 4096 tokens (original context boundary)
        ax1.axvline(4096, color="black", linestyle=":", linewidth=1.5,
                    label="4k boundary")

        ax1.set_xlabel("Generated tokens (absolute)", fontsize=12)
        ax1.set_ylabel(ylabel, fontsize=12)
        ax2.set_ylabel("# activations in bucket", fontsize=10, color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")
        ax1.set_title(title, fontsize=13)
        ax1.legend(loc="lower right", fontsize=9)
        ax1.set_xlim(left=0)
        ax1.set_ylim(bottom=0, top=1.0)
        ax1.grid(axis="y", alpha=0.3)

        out = figures_dir / fname
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info(f"Saved {out}")

    n_classes_mut = len(present_types)
    random_baseline_mut = 1.0 / max(n_classes_mut, 1)
    _safe_plot(
        "mut",
        f"mutation_type probe vs absolute token position (hard 16k, {n_classes_mut}-class)",
        "Val accuracy",
        random_baseline_mut,
        f"Random baseline ({random_baseline_mut:.2f})",
        "probe_hard_abstoken_mutation_type.png",
    )

    _safe_plot(
        "wbc",
        "will_be_correct probe vs absolute token position (hard 16k, buggy-only)",
        "Val accuracy",
        0.5,
        "Random baseline (0.50)",
        "probe_hard_abstoken_wbc.png",
    )

    logger.info("Plotting complete.")


if __name__ == "__main__":
    args = load_from_cli(ProbeAbsTokenArgs)
    run_probe_abstoken(args)
