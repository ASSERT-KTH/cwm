"""Overfit sanity test for the global probe.

Loads a tiny balanced subset (n_per_class examples per class) from the
SWEbench activations and attempts to overfit a probe to them with no
regularisation and a high learning rate.

A working training loop should drive loss to near 0 and accuracy to ~100%
within a few hundred epochs.  If it cannot, there is a code or data bug.

Usage:
    python -m interp.swerl.analysis.overfit_probe_test \\
        extract_dir=interp-swerl-extract n_per_class=8
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from cwm.common.params import load_from_cli
from interp.probes.train_probe import build_probe
from interp.swerl.analysis.run_probes import _collect_records, load_records

logger = logging.getLogger(__name__)


@dataclass
class OverfitArgs:
    extract_dir: str = "interp-swerl-extract"
    layer: int = 32
    n_per_class: int = 8       # examples per class in the tiny subset
    max_captures_per_file: int = 20
    n_epochs: int = 500
    lr: float = 1e-1           # high lr — we want to overfit fast
    architecture: str = "linear"
    hidden_dim: int = 256
    seed: int = 0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = load_from_cli(OverfitArgs)

    records = load_records(Path(args.extract_dir), layer=args.layer, max_captures_per_file=args.max_captures_per_file)
    all_acts, all_labels, _, _, _ = _collect_records(records, args.layer)

    # Pick n_per_class examples from each class
    pos_idx = [i for i, l in enumerate(all_labels) if l == 1]
    neg_idx = [i for i, l in enumerate(all_labels) if l == 0]
    assert len(pos_idx) >= args.n_per_class and len(neg_idx) >= args.n_per_class, \
        "Not enough examples per class"

    rng = torch.Generator().manual_seed(args.seed)
    pos_pick = torch.randperm(len(pos_idx), generator=rng)[:args.n_per_class].tolist()
    neg_pick = torch.randperm(len(neg_idx), generator=rng)[:args.n_per_class].tolist()
    chosen = [pos_idx[i] for i in pos_pick] + [neg_idx[i] for i in neg_pick]

    X = torch.stack([all_acts[i] for i in chosen]).float()  # [2*n, dim]
    y = torch.tensor([all_labels[i] for i in chosen], dtype=torch.long)

    # Standardise
    mu, std = X.mean(0, keepdim=True), X.std(0, keepdim=True).clamp(min=1e-6)
    X = (X - mu) / std

    logger.info(f"Tiny dataset: {X.shape[0]} examples, dim={X.shape[1]}, "
                f"labels={y.tolist()}")

    probe = build_probe(args.architecture, X.shape[1], 2, args.hidden_dim)
    # No weight decay — pure overfit test
    optimizer = torch.optim.Adam(probe.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.n_epochs):
        probe.train()
        logits = probe(X)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (logits.argmax(-1) == y).float().mean().item()
        if epoch % 50 == 0 or epoch == args.n_epochs - 1:
            logger.info(f"epoch {epoch:4d}  loss={loss.item():.4f}  acc={acc:.3f}")

    final_loss = loss.item()
    final_acc = (probe(X).argmax(-1) == y).float().mean().item()

    print(f"\n{'='*50}")
    print(f"Final loss: {final_loss:.4f}  (expected: near 0)")
    print(f"Final acc:  {final_acc:.3f}   (expected: 1.000)")
    if final_acc < 0.9:
        print("FAIL — probe cannot overfit tiny dataset. Bug in training loop or data.")
    else:
        print("PASS — probe can overfit. Training loop and data are correct.")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
