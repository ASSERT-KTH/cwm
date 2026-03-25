"""Logit lens applied across decode steps (temporal axis).

For each sample and layer, projects h_T[-1, L] through the unembedding head
at every (subsampled) decode step T. Produces:
  - P(correct_answer_token | T, L): the "commitment surface"
  - T*(L) = first step where P > threshold

This is the "free lunch" analysis: no labels required.

Usage:
    python -m interp.bug_trace.analysis.logit_lens_temporal \\
        traj_dir=./interp-bug-trajectories \\
        checkpoint_dir=./model_weights/cwm
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch

from cwm.common.params import load_from_cli
from interp.logit_lens.run_logit_lens import _load_norm_and_output, _rms_norm, apply_logit_lens

logger = logging.getLogger(__name__)


@dataclass
class LogitLensTemporalArgs:
    traj_dir: str = "interp-bug-trajectories"
    checkpoint_dir: str = "./model_weights/cwm"
    commit_threshold: float = 0.3  # P > threshold → "committed"
    top_k: int = 5
    layers: list[int] = field(default_factory=lambda: [16, 32, 48, 63])


def run_logit_lens_temporal(args: LogitLensTemporalArgs) -> None:
    logging.basicConfig(level=logging.INFO)

    traj_path = Path(args.traj_dir)
    index_path = traj_path / "index.jsonl"
    if not index_path.exists():
        raise FileNotFoundError(f"No index.jsonl in {args.traj_dir}")

    index: list[dict] = []
    with index_path.open() as f:
        for line in f:
            index.append(json.loads(line))

    norm_weight, output_weight, eps = _load_norm_and_output(args.checkpoint_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm_weight = norm_weight.to(device)
    output_weight = output_weight.to(device)

    traj_subdir = traj_path / "trajectories"
    results: list[dict] = []

    for meta in index:
        sid = meta["sample_id"]
        pt_path = traj_subdir / f"{sid}.pt"
        if not pt_path.exists():
            continue
        sample = torch.load(pt_path, map_location="cpu", weights_only=False)
        trajectory: dict = sample.get("trajectory", {})
        if not trajectory:
            continue

        # Determine target token IDs for correct_output
        correct_output = meta.get("correct_output", "")
        # We'll track P(any token that spells out the correct answer)
        # Rough: tokenize the repr of the correct output and use those token IDs
        # For now, skip target_token_ids (compute unconditional top-k only)

        step_results: dict[int, dict] = {}
        for layer_key, h_traj in trajectory.items():
            layer = int(layer_key)
            if layer not in args.layers:
                continue
            # h_traj: [T, dim], fp16
            h = h_traj.float().to(device)
            n_steps = h.shape[0]

            # Apply logit lens to all steps in one batch (faster than step-by-step)
            # h: [T, dim] → logits: [T, vocab] → top-k per step
            lens = apply_logit_lens(h, norm_weight, output_weight, eps, top_k=args.top_k)
            top1_probs = [row[0] if row else 0.0 for row in lens["top_k_probs"]]
            top1_ids = [row[0] if row else 0 for row in lens["top_k_ids"]]

            # Find commitment step T*: first T where top-1 prob > threshold
            t_star = next(
                (t for t, p in enumerate(top1_probs) if p > args.commit_threshold), None
            )

            step_results[layer] = {
                "top1_probs": top1_probs,
                "top1_ids": top1_ids,
                "t_star": t_star,
                "n_steps": n_steps,
            }

        results.append({
            "sample_id": sid,
            "is_buggy": meta.get("is_buggy", False),
            "correct": meta.get("correct", False),
            "mutation_type": meta.get("mutation_type", ""),
            "layers": step_results,
        })

    out_path = traj_path / "logit_lens_temporal.pt"
    torch.save(results, out_path)
    logger.info(f"Saved temporal logit lens for {len(results)} samples to {out_path}")

    # Summary: mean T* per layer, split by correct/incorrect and buggy/original
    print("\n=== Temporal Logit Lens Summary ===")
    for layer in args.layers:
        for is_buggy in [False, True]:
            t_stars = []
            for r in results:
                if r.get("is_buggy") != is_buggy:
                    continue
                layer_data = r.get("layers", {}).get(layer)
                if layer_data and layer_data.get("t_star") is not None:
                    t_stars.append(layer_data["t_star"])
            label = "buggy" if is_buggy else "original"
            if t_stars:
                mean_t = sum(t_stars) / len(t_stars)
                print(f"  Layer {layer:2d} [{label:8s}]: mean T* = {mean_t:.1f} (n={len(t_stars)})")


if __name__ == "__main__":
    args = load_from_cli(LogitLensTemporalArgs)
    run_logit_lens_temporal(args)
