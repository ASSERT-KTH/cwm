"""Visualisation utilities for bug-trace analysis results.

Produces publication-quality figures saved to traj_dir/figures/:
  - logit_lens_heatmap.png: P(commit) as a function of (T, L)
  - probe_heatmap.png: probe accuracy as a function of (T, L)
  - pca_trajectories.png: sample trajectories in PC1-PC2 space
  - change_point_hist.png: histogram of T* relative positions
  - ccs_separation.png: CCS separation accuracy per layer
  - dmd_spectrum.png: DMD eigenvalue spectrum

Usage:
    python -m interp.bug_trace.analysis.visualize \\
        traj_dir=./interp-bug-trajectories
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch

from cwm.common.params import load_from_cli

logger = logging.getLogger(__name__)


@dataclass
class VisualizeArgs:
    traj_dir: str = "interp-bug-trajectories"
    layers: list[int] = field(default_factory=lambda: [16, 32, 48, 63])
    n_traj_examples: int = 30      # trajectories to plot in PCA space
    dpi: int = 150


def _get_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        logger.warning("matplotlib not available; skipping plots")
        return None


def plot_probe_heatmap(traj_path: Path, target: str, layers: list[int], dpi: int) -> None:
    plt = _get_matplotlib()
    if plt is None:
        return
    probe_path = traj_path / f"probe_temporal_{target}.pt"
    if not probe_path.exists():
        logger.warning(f"No probe results at {probe_path}")
        return

    data = torch.load(probe_path, map_location="cpu", weights_only=False)
    heatmap = data["heatmap"]
    args_d = data.get("args", {})
    n_bins = args_d.get("n_time_bins", 10)

    import numpy as np
    grid = np.full((len(layers), n_bins), float("nan"))
    for li, layer in enumerate(layers):
        for b in range(n_bins):
            val = heatmap.get(layer, {}).get(b, {}).get("val_acc", float("nan"))
            grid[li, b] = val

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(grid, aspect="auto", vmin=0.4, vmax=1.0, cmap="RdYlGn")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([str(l) for l in layers])
    ax.set_xticks(range(n_bins))
    ax.set_xticklabels([f"{(b+0.5)/n_bins:.1f}" for b in range(n_bins)], rotation=45, ha="right")
    ax.set_xlabel("Relative time in generation (0=start, 1=end)")
    ax.set_ylabel("Layer")
    ax.set_title(f"Probe accuracy ({target}) — T × L heatmap")
    plt.colorbar(im, ax=ax, label="Val accuracy")
    plt.tight_layout()
    out = traj_path / "figures" / f"probe_heatmap_{target}.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=dpi)
    plt.close()
    logger.info(f"Saved probe heatmap to {out}")


def plot_pca_trajectories(traj_path: Path, layer: int, n_examples: int, dpi: int) -> None:
    plt = _get_matplotlib()
    if plt is None:
        return
    pca_path = traj_path / "pca_trajectory.pt"
    if not pca_path.exists():
        return

    data = torch.load(pca_path, map_location="cpu", weights_only=False)
    layer_data = data.get(layer)
    if layer_data is None:
        return

    traj_projs = layer_data.get("traj_projections", [])
    if not traj_projs:
        return

    import numpy as np
    import random
    rng = random.Random(42)

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = {"orig_correct": "#2196F3", "orig_wrong": "#90CAF9",
              "buggy_correct": "#F44336", "buggy_wrong": "#EF9A9A"}

    chosen = rng.sample(traj_projs, min(n_examples, len(traj_projs)))
    for item in chosen:
        proj = np.array(item["proj"])  # [T, 2]
        if proj.shape[0] < 2:
            continue
        is_buggy = item["is_buggy"]
        correct = item["correct"]
        if is_buggy and correct:
            c = colors["buggy_correct"]
        elif is_buggy:
            c = colors["buggy_wrong"]
        elif correct:
            c = colors["orig_correct"]
        else:
            c = colors["orig_wrong"]

        ax.plot(proj[:, 0], proj[:, 1], alpha=0.3, color=c, linewidth=0.8)
        # Arrow at end
        if proj.shape[0] >= 2:
            ax.annotate(
                "", xy=proj[-1], xytext=proj[-2],
                arrowprops=dict(arrowstyle="->", color=c, lw=0.8),
            )

    # Legend
    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], color=colors["orig_correct"], label="original, correct"),
        Line2D([0], [0], color=colors["orig_wrong"], label="original, incorrect"),
        Line2D([0], [0], color=colors["buggy_correct"], label="buggy, correct"),
        Line2D([0], [0], color=colors["buggy_wrong"], label="buggy, incorrect"),
    ]
    ax.legend(handles=legend, fontsize=8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"Representation trajectories in PCA space (layer {layer})")
    plt.tight_layout()
    out = traj_path / "figures" / f"pca_trajectories_L{layer}.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=dpi)
    plt.close()
    logger.info(f"Saved PCA trajectory plot to {out}")


def plot_change_point_hist(traj_path: Path, layers: list[int], method: str, dpi: int) -> None:
    plt = _get_matplotlib()
    if plt is None:
        return
    cp_path = traj_path / f"change_points_{method}.pt"
    if not cp_path.exists():
        return

    results = torch.load(cp_path, map_location="cpu", weights_only=False)

    import numpy as np
    fig, axes = plt.subplots(1, len(layers), figsize=(4 * len(layers), 4), sharey=False)
    if len(layers) == 1:
        axes = [axes]

    for ax, layer in zip(axes, layers):
        orig_rels = [r["layers"][layer]["t_star_relative"]
                     for r in results if layer in r.get("layers", {}) and not r.get("is_buggy")]
        bug_rels = [r["layers"][layer]["t_star_relative"]
                    for r in results if layer in r.get("layers", {}) and r.get("is_buggy")]

        bins = np.linspace(0, 1, 21)
        if orig_rels:
            ax.hist(orig_rels, bins=bins, alpha=0.6, label="original", color="#2196F3")
        if bug_rels:
            ax.hist(bug_rels, bins=bins, alpha=0.6, label="buggy", color="#F44336")
        ax.set_xlabel("Relative T* (0=start, 1=end)")
        ax.set_title(f"Layer {layer}")
        ax.legend(fontsize=7)

    fig.suptitle(f"Change point T* distribution ({method})")
    plt.tight_layout()
    out = traj_path / "figures" / f"change_point_hist_{method}.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=dpi)
    plt.close()
    logger.info(f"Saved change point histogram to {out}")


def plot_probe_content_heatmap(traj_path: Path, layers: list[int], dpi: int) -> None:
    plt = _get_matplotlib()
    if plt is None:
        return
    probe_path = traj_path / "probe_content_mutation_type.pt"
    if not probe_path.exists():
        logger.warning(f"No probe_content results at {probe_path}")
        return

    data = torch.load(probe_path, map_location="cpu", weights_only=False)
    heatmap = data["heatmap"]
    mutation_types = data.get("mutation_types", [])
    majority_bl = data.get("majority_baseline", 0.35)
    random_bl = data.get("random_baseline", 0.2)
    args_d = data.get("args", {})
    n_bins = args_d.get("n_time_bins", 10)

    import numpy as np
    grid = np.full((len(layers), n_bins), float("nan"))
    perm_bls = []
    for li, layer in enumerate(layers):
        perm_bls.append(heatmap.get(layer, {}).get("perm_baseline", float("nan")))
        for b in range(n_bins):
            val = heatmap.get(layer, {}).get(b, {}).get("val_acc", float("nan"))
            grid[li, b] = val

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(grid, aspect="auto", vmin=0.1, vmax=0.8, cmap="RdYlGn")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([str(l) for l in layers])
    ax.set_xticks(range(n_bins))
    ax.set_xticklabels([f"{(b+0.5)/n_bins:.1f}" for b in range(n_bins)], rotation=45, ha="right")
    ax.set_xlabel("Relative time in generation (0=start, 1=end)")
    ax.set_ylabel("Layer")
    title = (
        f"mutation_type probe ({len(mutation_types)}-class) — T × L heatmap\n"
        f"random={random_bl:.2f}  majority={majority_bl:.2f}  "
        f"perm_bl={[f'{p:.2f}' for p in perm_bls]}"
    )
    ax.set_title(title, fontsize=8)
    plt.colorbar(im, ax=ax, label="Val accuracy")
    plt.tight_layout()
    out = traj_path / "figures" / "probe_heatmap_mutation_type.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=dpi)
    plt.close()
    logger.info(f"Saved mutation_type probe heatmap to {out}")


def plot_pca_mutation_type(traj_path: Path, layer: int, n_examples: int, dpi: int) -> None:
    """PCA scatter colored by mutation_type (first decode step only)."""
    plt = _get_matplotlib()
    if plt is None:
        return
    pca_path = traj_path / "pca_trajectory.pt"
    if not pca_path.exists():
        return

    data = torch.load(pca_path, map_location="cpu", weights_only=False)
    layer_data = data.get(layer)
    if layer_data is None:
        return

    traj_projs = layer_data.get("traj_projections", [])
    if not traj_projs:
        return

    import numpy as np

    mutation_colors = {
        "condition_flip": "#E91E63",
        "off_by_one_minus": "#FF9800",
        "off_by_one_plus": "#FFC107",
        "wrong_comparator": "#9C27B0",
        "wrong_operator": "#00BCD4",
        "none": "#9E9E9E",
    }

    fig, ax = plt.subplots(figsize=(8, 8))
    seen_types: set = set()
    for item in traj_projs:
        proj = np.array(item["proj"])  # [T, 2]
        if proj.shape[0] < 1:
            continue
        mt = item.get("mutation_type", "none") or "none"
        color = mutation_colors.get(mt, "#9E9E9E")
        # Plot only first step as a scatter point
        ax.scatter(proj[0, 0], proj[0, 1], c=color, s=8, alpha=0.5, label=(mt if mt not in seen_types else ""))
        seen_types.add(mt)

    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=col, markersize=8, label=mt)
        for mt, col in mutation_colors.items() if mt in seen_types
    ]
    ax.legend(handles=legend, fontsize=8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"PCA scatter by mutation_type (layer {layer}, first decode step)")
    plt.tight_layout()
    out = traj_path / "figures" / f"pca_mutation_type_L{layer}.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=dpi)
    plt.close()
    logger.info(f"Saved mutation_type PCA plot to {out}")


def plot_ccs_separation(traj_path: Path, dpi: int) -> None:
    plt = _get_matplotlib()
    if plt is None:
        return
    ccs_path = traj_path / "ccs_mean.pt"
    if not ccs_path.exists():
        return

    data = torch.load(ccs_path, map_location="cpu", weights_only=False)
    results = data.get("results", {})
    if not results:
        return

    import numpy as np
    layers = sorted(results.keys())
    accs = [results[l].get("separation_acc", 0) for l in layers]
    losses = [results[l].get("loss", 0) for l in layers]

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()
    ax1.bar(range(len(layers)), accs, color="#2196F3", alpha=0.7, label="Separation accuracy")
    ax2.plot(range(len(layers)), losses, color="#F44336", marker="o", label="CCS loss")
    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels([str(l) for l in layers])
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Separation accuracy", color="#2196F3")
    ax2.set_ylabel("CCS loss", color="#F44336")
    ax1.set_title("CCS: direction separating original vs buggy representations")
    ax1.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    plt.tight_layout()
    out = traj_path / "figures" / "ccs_separation.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=dpi)
    plt.close()
    logger.info(f"Saved CCS plot to {out}")


def plot_dmd_spectrum(traj_path: Path, layers: list[int], dpi: int) -> None:
    plt = _get_matplotlib()
    if plt is None:
        return
    dmd_path = traj_path / "dmd.pt"
    if not dmd_path.exists():
        return

    data = torch.load(dmd_path, map_location="cpu", weights_only=False)
    results = data.get("results", {})

    import numpy as np
    fig, axes = plt.subplots(1, len(layers), figsize=(4 * len(layers), 4))
    if len(layers) == 1:
        axes = [axes]

    for ax, layer in zip(axes, layers):
        layer_data = results.get(layer, {})
        dmd_all = layer_data.get("dmd_all", {})
        eigvals = dmd_all.get("eigenvalues")
        if eigvals is None:
            continue
        eigvals = eigvals.numpy()
        # Unit circle
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.3, lw=0.8)
        ax.scatter(eigvals.real, eigvals.imag, s=20, alpha=0.8, c="steelblue")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlabel("Re(λ)")
        ax.set_ylabel("Im(λ)")
        ax.set_title(f"Layer {layer}")
        ax.set_aspect("equal")

    fig.suptitle("DMD eigenvalue spectrum (persistence of temporal modes)")
    plt.tight_layout()
    out = traj_path / "figures" / "dmd_spectrum.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=dpi)
    plt.close()
    logger.info(f"Saved DMD spectrum to {out}")


def run_visualize(args: VisualizeArgs) -> None:
    logging.basicConfig(level=logging.INFO)
    traj_path = Path(args.traj_dir)
    (traj_path / "figures").mkdir(parents=True, exist_ok=True)

    plot_probe_heatmap(traj_path, "is_buggy", args.layers, args.dpi)
    plot_probe_heatmap(traj_path, "will_be_correct", args.layers, args.dpi)
    plot_probe_content_heatmap(traj_path, args.layers, args.dpi)
    for layer in args.layers:
        plot_pca_trajectories(traj_path, layer, args.n_traj_examples, args.dpi)
        plot_pca_mutation_type(traj_path, layer, args.n_traj_examples, args.dpi)
    plot_change_point_hist(traj_path, args.layers, "cosine", args.dpi)
    plot_change_point_hist(traj_path, args.layers, "l2", args.dpi)
    plot_ccs_separation(traj_path, args.dpi)
    plot_dmd_spectrum(traj_path, args.layers, args.dpi)
    logger.info(f"All figures saved to {traj_path}/figures/")


if __name__ == "__main__":
    args = load_from_cli(VisualizeArgs)
    run_visualize(args)
