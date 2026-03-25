"""Change-point detection on decode-step trajectories.

Finds T* — the step where each trajectory undergoes its largest qualitative shift.
Correlates T* with the generated text at that position to find "aha moments".

Two metrics for detecting change points:
  1. L2 distance between consecutive steps: ||h_{T+1} - h_T||
  2. Cosine distance: 1 - cos(h_T, h_{T+1})

Usage:
    python -m interp.bug_trace.analysis.change_point \\
        traj_dir=./interp-bug-trajectories \\
        layer=32
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
class ChangePointArgs:
    traj_dir: str = "interp-bug-trajectories"
    layers: list[int] = field(default_factory=lambda: [16, 32, 48, 63])
    method: str = "cosine"   # l2 | cosine
    context_window: int = 3  # number of decoded tokens around T* to show


def _cosine_distances(h: torch.Tensor) -> torch.Tensor:
    """Compute cosine distance between consecutive steps. Shape [T-1]."""
    h_norm = torch.nn.functional.normalize(h.float(), dim=-1)
    dots = (h_norm[:-1] * h_norm[1:]).sum(-1)
    return 1.0 - dots


def _l2_distances(h: torch.Tensor) -> torch.Tensor:
    """L2 distance between consecutive steps. Shape [T-1]."""
    return (h[1:].float() - h[:-1].float()).norm(dim=-1)


def _find_change_points(h: torch.Tensor, method: str) -> tuple[torch.Tensor, int]:
    """Return distance series and argmax (the change point step)."""
    if h.shape[0] < 2:
        return torch.zeros(1), 0
    if method == "cosine":
        dists = _cosine_distances(h)
    else:
        dists = _l2_distances(h)
    t_star = int(dists.argmax().item())
    return dists, t_star


def run_change_point(args: ChangePointArgs) -> None:
    logging.basicConfig(level=logging.INFO)

    traj_path = Path(args.traj_dir)
    index: list[dict] = []
    with (traj_path / "index.jsonl").open() as f:
        for line in f:
            index.append(json.loads(line))
    traj_subdir = traj_path / "trajectories"

    all_results: list[dict] = []

    for meta in index:
        sid = meta["sample_id"]
        pt = traj_subdir / f"{sid}.pt"
        if not pt.exists():
            continue
        sample = torch.load(pt, map_location="cpu", weights_only=False)
        traj = sample.get("trajectory", {})
        decode_token_ids = sample.get("decode_token_ids", [])
        stride = sample.get("stride", 1)
        generated_text = sample.get("generated_text", "")

        layer_cps: dict = {}
        for layer in args.layers:
            h = traj.get(layer)
            if h is None:
                h = traj.get(str(layer))
            if h is None or h.shape[0] < 2:
                continue
            dists, t_star = _find_change_points(h, args.method)

            # Map t_star (in strided trajectory space) to actual token position
            actual_token_idx = t_star * stride

            # Surrounding context in token IDs
            ctx_start = max(0, actual_token_idx - args.context_window)
            ctx_end = min(len(decode_token_ids), actual_token_idx + args.context_window + 1)
            ctx_token_ids = decode_token_ids[ctx_start:ctx_end]

            layer_cps[layer] = {
                "t_star": t_star,
                "t_star_actual": actual_token_idx,
                "t_star_relative": t_star / max(h.shape[0] - 1, 1),
                "max_distance": float(dists.max().item()),
                "mean_distance": float(dists.mean().item()),
                "distances": dists.tolist(),
                "context_token_ids": ctx_token_ids,
            }

        all_results.append({
            "sample_id": sid,
            "is_buggy": meta.get("is_buggy", False),
            "correct": meta.get("correct", False),
            "mutation_type": meta.get("mutation_type", ""),
            "pair_id": meta.get("pair_id", ""),
            "layers": layer_cps,
        })

    out = traj_path / f"change_points_{args.method}.pt"
    torch.save(all_results, out)
    logger.info(f"Saved change point results to {out}")

    # Summary: mean T* relative position per condition
    print(f"\n=== Change Point Summary (method={args.method}) ===")
    for layer in args.layers:
        for is_buggy in [False, True]:
            t_rels = []
            for r in all_results:
                if r.get("is_buggy") != is_buggy:
                    continue
                cp = r.get("layers", {}).get(layer)
                if cp:
                    t_rels.append(cp["t_star_relative"])
            label = "buggy" if is_buggy else "original"
            if t_rels:
                mean_rel = sum(t_rels) / len(t_rels)
                print(
                    f"  Layer {layer:2d} [{label:8s}]: "
                    f"mean T*/T = {mean_rel:.3f} (n={len(t_rels)})"
                )

    # Show a few examples with context
    print("\n=== Example Change Points (layer 32) ===")
    shown = 0
    for r in all_results:
        if shown >= 5:
            break
        cp = r.get("layers", {}).get(32)
        if cp is None:
            continue
        print(
            f"\n  {r['sample_id']} [{'buggy' if r['is_buggy'] else 'orig'},"
            f" {r['mutation_type']}, correct={r['correct']}]"
        )
        print(f"  T* = {cp['t_star_actual']} (rel={cp['t_star_relative']:.2f}), "
              f"max_dist={cp['max_distance']:.4f}")
        print(f"  Context token IDs: {cp['context_token_ids']}")
        shown += 1


if __name__ == "__main__":
    args = load_from_cli(ChangePointArgs)
    run_change_point(args)
