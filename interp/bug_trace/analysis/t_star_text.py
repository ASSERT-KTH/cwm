"""T* text correlation: find where the largest representational shift occurs in text.

For each trajectory:
  1. Compute inter-step cosine distance: dist[t] = 1 - cos_sim(h_t, h_{t-1})
  2. Find T* = argmax(dist) — the step with the largest representational shift
  3. Identify what token was generated at T* from decode_token_ids

Aggregate: frequency distribution of T* tokens across samples.
Per-sample: plot distance curve vs. relative position, with T* highlighted.

Outputs:
  {traj_dir}/t_star_text.pt         — per-sample T* data
  {traj_dir}/figures/t_star_dist_{layer}.png  — aggregate distance profile
  {traj_dir}/figures/t_star_samples_{layer}/  — per-sample figures (subset)

Usage:
    python -m interp.bug_trace.analysis.t_star_text \\
        traj_dir=./interp-bug-trajectories \\
        checkpoint_dir=./model_weights/cwm
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import torch

from cwm.common.params import load_from_cli

logger = logging.getLogger(__name__)


@dataclass
class TStarTextArgs:
    traj_dir: str = "interp-bug-trajectories"
    layers: list[int] = field(default_factory=lambda: [16, 32, 48, 63])
    checkpoint_dir: str = "./model_weights/cwm"
    n_sample_figures: int = 100   # per-sample figures to generate (random subset)
    dpi: int = 120
    seed: int = 42


def _cos_dist_profile(h_traj: torch.Tensor) -> torch.Tensor:
    """Compute 1 - cos_sim(h_t, h_{t-1}) for t=1..T-1. Returns [T-1] tensor."""
    h = h_traj.float()
    h_norm = h / (h.norm(dim=1, keepdim=True).clamp(min=1e-8))
    cos_sim = (h_norm[1:] * h_norm[:-1]).sum(dim=1)  # [T-1]
    return 1.0 - cos_sim


def _get_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        logger.warning("matplotlib not available; skipping plots")
        return None


def _plot_sample(
    plt,
    dist: torch.Tensor,       # [T-1]
    t_star: int,               # index in dist
    token_strs: list[str],     # decoded token at each generation step (strided)
    sample_id: str,
    is_buggy: bool,
    mutation_type: str,
    layer: int,
    out_path: Path,
    dpi: int,
) -> None:
    T = len(dist)
    x = [i / max(T - 1, 1) for i in range(T)]
    color = "#F44336" if is_buggy else "#2196F3"

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(x, dist.numpy(), color=color, linewidth=0.8, alpha=0.8)
    ax.axvline(x=x[t_star], color="black", linewidth=1.5, linestyle="--", label=f"T*={x[t_star]:.2f}")

    # Annotate T* with the token text
    if 0 <= t_star < len(token_strs):
        tok_str = repr(token_strs[t_star])[:30]
        ax.annotate(
            tok_str, xy=(x[t_star], dist[t_star].item()),
            xytext=(x[t_star] + 0.02, dist[t_star].item()),
            fontsize=7, color="black",
            arrowprops=dict(arrowstyle="->", color="black", lw=0.5),
        )

    # X-axis ticks: show token text at evenly spaced positions
    n_ticks = min(20, T)
    tick_idx = [int(i * (T - 1) / max(n_ticks - 1, 1)) for i in range(n_ticks)]
    ax.set_xticks([x[i] for i in tick_idx])
    ax.set_xticklabels(
        [repr(token_strs[i])[:8] if i < len(token_strs) else "" for i in tick_idx],
        rotation=45, ha="right", fontsize=6,
    )
    title = f"{sample_id} | layer {layer} | {'buggy:' + mutation_type if is_buggy else 'original'}"
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("Relative position in generation")
    ax.set_ylabel("Inter-step cosine distance")
    ax.legend(fontsize=7)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def run_t_star_text(args: TStarTextArgs) -> None:
    logging.basicConfig(level=logging.INFO)
    rng = random.Random(args.seed)

    traj_path = Path(args.traj_dir)
    index: list[dict] = []
    with (traj_path / "index.jsonl").open() as f:
        for line in f:
            index.append(json.loads(line))
    traj_subdir = traj_path / "trajectories"

    # Load tokenizer for decoding token IDs
    tokenizer = None
    try:
        from cwm.fastgen.utils.loading import build_tokenizer_from_ckpt
        tokenizer = build_tokenizer_from_ckpt(args.checkpoint_dir)
        logger.info("Tokenizer loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load tokenizer: {e}. Token text will be shown as IDs.")

    plt = _get_matplotlib()

    # Choose samples for per-sample figures
    fig_sample_ids = set(
        m["sample_id"] for m in rng.sample(index, min(args.n_sample_figures, len(index)))
    )

    # Per-sample results
    all_results: list[dict] = []

    # Aggregate: per-layer distance profiles and T* token counters
    dist_profiles: dict[int, list[torch.Tensor]] = {L: [] for L in args.layers}
    t_star_relpos: dict[int, dict] = {L: {"buggy": [], "original": []} for L in args.layers}
    t_star_tokens: dict[int, Counter] = {L: Counter() for L in args.layers}

    for meta in index:
        sid = meta["sample_id"]
        pt = traj_subdir / f"{sid}.pt"
        if not pt.exists():
            continue
        sample = torch.load(pt, map_location="cpu", weights_only=False)
        traj = sample.get("trajectory", {})
        decode_ids = sample.get("decode_token_ids", [])
        is_buggy = meta.get("is_buggy", False)
        mutation_type = meta.get("mutation_type", "none")

        # Decode token IDs to strings
        token_strs: list[str] = []
        if tokenizer is not None and decode_ids:
            for tid in decode_ids:
                try:
                    token_strs.append(tokenizer.decode([tid]))
                except Exception:
                    token_strs.append(f"<{tid}>")
        else:
            token_strs = [f"<{tid}>" for tid in decode_ids]

        sample_result: dict = {
            "sample_id": sid,
            "is_buggy": is_buggy,
            "mutation_type": mutation_type,
            "correct": meta.get("correct", False),
            "t_star": {},
        }

        for layer in args.layers:
            h = traj.get(layer)
            if h is None:
                h = traj.get(str(layer))
            if h is None or h.shape[0] < 2:
                continue

            dist = _cos_dist_profile(h)   # [T-1]
            T = len(dist)
            t_star_idx = int(dist.argmax().item())
            t_star_rel = t_star_idx / max(T - 1, 1)

            sample_result["t_star"][layer] = {
                "t_star_idx": t_star_idx,
                "t_star_relative": t_star_rel,
                "t_star_dist": float(dist[t_star_idx].item()),
                "t_star_token_id": decode_ids[t_star_idx] if t_star_idx < len(decode_ids) else None,
                "t_star_token_str": token_strs[t_star_idx] if t_star_idx < len(token_strs) else "",
            }

            # Aggregate
            dist_profiles[layer].append(dist)
            cond = "buggy" if is_buggy else "original"
            t_star_relpos[layer][cond].append(t_star_rel)
            if t_star_idx < len(token_strs):
                t_star_tokens[layer][token_strs[t_star_idx]] += 1

            # Per-sample figure
            if plt is not None and sid in fig_sample_ids:
                out_fig = (
                    traj_path / "figures" / f"t_star_samples_L{layer}" / f"{sid}.png"
                )
                _plot_sample(
                    plt, dist, t_star_idx, token_strs, sid,
                    is_buggy, mutation_type, layer, out_fig, args.dpi,
                )

        all_results.append(sample_result)

    # Aggregate figures: mean distance profile + T* histogram per layer
    if plt is not None:
        import numpy as np

        for layer in args.layers:
            profiles = dist_profiles[layer]
            if not profiles:
                continue

            fig_dir = traj_path / "figures"
            fig_dir.mkdir(exist_ok=True)

            # Interpolate all profiles to a common grid and plot mean ± std
            grid = np.linspace(0, 1, 100)
            interp_profiles = []
            for d in profiles:
                T = len(d)
                xs = np.linspace(0, 1, T)
                interp_profiles.append(np.interp(grid, xs, d.numpy()))
            arr = np.stack(interp_profiles)  # [N, 100]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

            # Mean profile
            ax1.fill_between(grid, arr.mean(0) - arr.std(0), arr.mean(0) + arr.std(0),
                             alpha=0.3, color="steelblue")
            ax1.plot(grid, arr.mean(0), color="steelblue", linewidth=1.5, label="All")
            # Split by buggy/original
            for cond, color in [("buggy", "#F44336"), ("original", "#2196F3")]:
                idxs = [i for i, m in enumerate(all_results)
                        if layer in m["t_star"] and (m["is_buggy"] == (cond == "buggy"))]
                if idxs:
                    sub = arr[idxs]
                    ax1.plot(grid, sub.mean(0), color=color, linewidth=1, linestyle="--",
                             label=cond, alpha=0.8)
            ax1.set_xlabel("Relative position")
            ax1.set_ylabel("Inter-step cosine distance")
            ax1.set_title(f"Layer {layer}: Mean distance profile ± 1 std")
            ax1.legend(fontsize=8)

            # T* histogram
            buggy_tstars = t_star_relpos[layer]["buggy"]
            orig_tstars = t_star_relpos[layer]["original"]
            bins = np.linspace(0, 1, 21)
            if buggy_tstars:
                ax2.hist(buggy_tstars, bins=bins, alpha=0.6, label="buggy", color="#F44336")
            if orig_tstars:
                ax2.hist(orig_tstars, bins=bins, alpha=0.6, label="original", color="#2196F3")
            ax2.set_xlabel("T* relative position")
            ax2.set_ylabel("Count")
            ax2.set_title(f"Layer {layer}: T* distribution")
            ax2.legend(fontsize=8)

            plt.tight_layout()
            out = fig_dir / f"t_star_dist_L{layer}.png"
            plt.savefig(out, dpi=args.dpi)
            plt.close()
            logger.info(f"Saved T* aggregate figure to {out}")

    # Print top-20 T* tokens per layer
    for layer in args.layers:
        top20 = t_star_tokens[layer].most_common(20)
        if top20:
            print(f"\nLayer {layer}: top-20 tokens at T* (n_samples={sum(t_star_tokens[layer].values())})")
            for tok, cnt in top20:
                print(f"  {repr(tok):<30} {cnt:>4}")

    # Save results
    out = traj_path / "t_star_text.pt"
    torch.save({
        "results": all_results,
        "t_star_relpos": t_star_relpos,
        "t_star_top_tokens": {L: t_star_tokens[L].most_common(50) for L in args.layers},
        "args": vars(args),
    }, out)
    logger.info(f"Saved T* text results to {out}")


if __name__ == "__main__":
    args = load_from_cli(TStarTextArgs)
    run_t_star_text(args)
