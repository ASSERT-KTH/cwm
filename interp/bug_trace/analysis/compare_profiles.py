"""Side-by-side comparison of mutation_type probe temporal profiles: easy vs hard.

Easy mutations (off_by_one, condition_flip, wrong_comparator, wrong_operator) are
visible at the syntactic level — the model can read the mutation type from the prompt
code at token 0.  A flat temporal profile indicates prompt-reading.

Hard mutations (wrong_variable, deleted_accumulator, swapped_arguments) require
execution reasoning — the model must mentally simulate the program to identify what
changed.  A rising temporal profile indicates active computation.

This script loads the probe_content_mutation_type.pt from both easy and hard
trajectory directories and produces:
  1. A side-by-side heatmap (layer × time_bin) for each group.
  2. Per-layer line plots of bin0 vs bin9 accuracy for easy and hard.
  3. A summary table comparing flat vs rising profiles.

Usage:
    python -m interp.bug_trace.analysis.compare_profiles \
        easy_traj_dir=./interp-bug-trajectories-track_a \
        hard_traj_dir=./interp-bug-trajectories-hard
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch

from cwm.common.params import load_from_cli

logger = logging.getLogger(__name__)


@dataclass
class CompareArgs:
    easy_traj_dir: str = "./interp-bug-trajectories-track_a"
    hard_traj_dir: str = "./interp-bug-trajectories-hard"
    layers: list[int] = field(default_factory=lambda: [16, 32, 48, 63])
    n_time_bins: int = 10


def _load_heatmap(traj_dir: str) -> dict | None:
    """Load probe_content_mutation_type.pt from traj_dir. Returns heatmap dict or None."""
    path = Path(traj_dir) / "probe_content_mutation_type.pt"
    if not path.exists():
        logger.warning(f"Not found: {path}")
        return None
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data.get("heatmap", {})


def _print_heatmap_table(label: str, heatmap: dict, layers: list[int], n_bins: int) -> None:
    print(f"\n=== {label} — mutation_type probe accuracy (layer × bin) ===")
    print(f"{'Layer':>6}", end="")
    for b in range(n_bins):
        print(f"  bin{b:02d}", end="")
    print("  | perm_bl  rise")
    for layer in layers:
        if layer not in heatmap:
            continue
        row = heatmap[layer]
        perm_bl = row.get("perm_baseline", float("nan"))
        accs = []
        for b in range(n_bins):
            v = row.get(b, {})
            accs.append(v.get("val_acc", float("nan")) if isinstance(v, dict) else float("nan"))
        # Rise: bin9 - bin0
        rise = accs[-1] - accs[0] if (accs[0] == accs[0] and accs[-1] == accs[-1]) else float("nan")
        print(f"{layer:>6}", end="")
        for acc in accs:
            if acc != acc:
                print("     nan", end="")
            else:
                print(f"  {acc:.3f}", end="")
        if rise != rise:
            print(f"  | {perm_bl:.3f}   nan")
        else:
            print(f"  | {perm_bl:.3f}  {rise:+.3f}")


def _compute_rise(heatmap: dict, layer: int, n_bins: int) -> float:
    """bin9 accuracy - bin0 accuracy at a given layer."""
    row = heatmap.get(layer, {})
    b0 = row.get(0, {})
    b9 = row.get(n_bins - 1, {})
    acc0 = b0.get("val_acc", float("nan")) if isinstance(b0, dict) else float("nan")
    acc9 = b9.get("val_acc", float("nan")) if isinstance(b9, dict) else float("nan")
    if acc0 != acc0 or acc9 != acc9:
        return float("nan")
    return acc9 - acc0


def run_compare(args: CompareArgs) -> None:
    logging.basicConfig(level=logging.INFO)

    hm_easy = _load_heatmap(args.easy_traj_dir)
    hm_hard = _load_heatmap(args.hard_traj_dir)

    if hm_easy is None and hm_hard is None:
        logger.error("Neither easy nor hard probe results found. Run probe_content first.")
        return

    if hm_easy is not None:
        _print_heatmap_table("EASY mutations", hm_easy, args.layers, args.n_time_bins)

    if hm_hard is not None:
        _print_heatmap_table("HARD mutations", hm_hard, args.layers, args.n_time_bins)

    # Summary comparison table
    if hm_easy is not None and hm_hard is not None:
        print("\n=== Summary: temporal rise (bin9 - bin0) by layer ===")
        print(f"{'Layer':>6}  {'Easy rise':>10}  {'Hard rise':>10}  {'Delta (H-E)':>12}  {'Interpretation'}")
        print("-" * 70)
        for layer in args.layers:
            rise_easy = _compute_rise(hm_easy, layer, args.n_time_bins)
            rise_hard = _compute_rise(hm_hard, layer, args.n_time_bins)
            if rise_easy != rise_easy or rise_hard != rise_hard:
                interp = "missing data"
                delta = float("nan")
            else:
                delta = rise_hard - rise_easy
                if rise_hard > 0.05 and rise_easy < 0.03:
                    interp = "RISING in hard only → active reasoning signal"
                elif rise_hard > 0.05 and rise_easy > 0.03:
                    interp = "both rising → may still be prompt artifact"
                elif rise_hard < 0.01 and rise_easy < 0.01:
                    interp = "both flat → prompt-reading artifact in both"
                elif rise_hard < 0.01:
                    interp = "flat in hard → no active reasoning signal"
                else:
                    interp = "ambiguous"
            if delta != delta:
                print(f"{layer:>6}  {rise_easy:>+10.3f}  {rise_hard:>+10.3f}  {'nan':>12}  {interp}")
            else:
                print(f"{layer:>6}  {rise_easy:>+10.3f}  {rise_hard:>+10.3f}  {delta:>+12.3f}  {interp}")

        # Try to save a figure if matplotlib is available
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
            n_bins = args.n_time_bins
            x = list(range(n_bins))

            for ax, (hm, label) in zip(axes, [(hm_easy, "Easy mutations"), (hm_hard, "Hard mutations")]):
                for layer in args.layers:
                    row = hm.get(layer, {})
                    accs = []
                    for b in range(n_bins):
                        v = row.get(b, {})
                        accs.append(v.get("val_acc", float("nan")) if isinstance(v, dict) else float("nan"))
                    ax.plot(x, accs, marker="o", label=f"L{layer}")
                # Random baseline
                perm_bl_vals = [hm.get(layer, {}).get("perm_baseline", float("nan")) for layer in args.layers]
                perm_bl = next((v for v in perm_bl_vals if v == v), float("nan"))
                if perm_bl == perm_bl:
                    ax.axhline(perm_bl, color="gray", linestyle="--", alpha=0.5, label="perm baseline")
                ax.set_title(label)
                ax.set_xlabel("Time bin")
                ax.set_ylabel("Val accuracy")
                ax.legend(fontsize=8)
                ax.set_ylim(0, 1)
                ax.grid(alpha=0.3)

            fig.suptitle("Temporal probe profile: easy vs hard mutations", fontsize=13)
            plt.tight_layout()

            out_dir = Path(args.hard_traj_dir) / "figures"
            out_dir.mkdir(parents=True, exist_ok=True)
            out = out_dir / "compare_profiles_easy_vs_hard.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved comparison figure to {out}")
            print(f"\nFigure saved: {out}")

        except Exception as e:
            logger.warning(f"Could not save figure: {e}")


if __name__ == "__main__":
    args = load_from_cli(CompareArgs)
    run_compare(args)
