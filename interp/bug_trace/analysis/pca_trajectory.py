"""PCA and functional PCA on decode-step trajectories.

Finds the dominant directions of variation across all trajectories.
Checks whether PC1 separates buggy/original or correct/incorrect generations.

Also computes per-sample trajectory in PC space and exports for visualisation.

Usage:
    python -m interp.bug_trace.analysis.pca_trajectory \\
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
class PCATrajectoryArgs:
    traj_dir: str = "interp-bug-trajectories"
    layers: list[int] = field(default_factory=lambda: [16, 32, 48, 63])
    n_components: int = 32
    max_samples_for_fit: int = 500   # subsample for PCA fitting (memory)
    seed: int = 42


def _load_flat_matrix(
    index: list[dict],
    traj_subdir: Path,
    layer: int,
    max_samples: int,
    rng_seed: int,
) -> tuple[torch.Tensor, list[dict]]:
    """Return X [N, dim] and metadata list, sampling at most max_samples trajectories."""
    import random
    rng = random.Random(rng_seed)
    chosen = list(range(len(index)))
    if len(chosen) > max_samples:
        chosen = rng.sample(chosen, max_samples)

    vectors = []
    metas = []
    for i in chosen:
        meta = index[i]
        sid = meta["sample_id"]
        pt = traj_subdir / f"{sid}.pt"
        if not pt.exists():
            continue
        sample = torch.load(pt, map_location="cpu", weights_only=False)
        traj = sample.get("trajectory", {})
        h = traj.get(layer)
        if h is None:
            h = traj.get(str(layer))
        if h is None or h.shape[0] == 0:
            continue
        # Use mean-pooled representation for PCA fitting
        vectors.append(h.float().mean(0))
        metas.append(meta)

    if not vectors:
        return torch.zeros(0, 1), []
    return torch.stack(vectors), metas


def run_pca_trajectory(args: PCATrajectoryArgs) -> None:
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(args.seed)

    traj_path = Path(args.traj_dir)
    index: list[dict] = []
    with (traj_path / "index.jsonl").open() as f:
        for line in f:
            index.append(json.loads(line))
    traj_subdir = traj_path / "trajectories"

    all_layer_results: dict = {}

    for layer in args.layers:
        logger.info(f"PCA: layer {layer}")

        # Fit PCA on mean-pooled representations
        X, metas = _load_flat_matrix(
            index, traj_subdir, layer, args.max_samples_for_fit, args.seed
        )
        if X.shape[0] < 10:
            logger.warning(f"Layer {layer}: not enough data ({X.shape[0]} samples), skipping")
            continue

        # PCA via SVD (on centred data)
        mu = X.mean(0, keepdim=True)
        X_c = X - mu
        U, S, Vh = torch.linalg.svd(X_c, full_matrices=False)
        # Vh: [n_comp, dim] — principal component directions
        components = Vh[: args.n_components]  # [K, dim]
        explained_var_ratio = (S[: args.n_components] ** 2) / (S ** 2).sum()

        # Check if PC1/PC2 separates is_buggy
        labels_buggy = torch.tensor([int(m.get("is_buggy", False)) for m in metas])
        labels_correct = torch.tensor([int(m.get("correct", False)) for m in metas])

        projections = X_c @ components.T  # [N, K]

        # Point-biserial correlation of PC1 with is_buggy
        def _pbc(proj_col: torch.Tensor, labels: torch.Tensor) -> float:
            """Point-biserial correlation."""
            pos = proj_col[labels == 1]
            neg = proj_col[labels == 0]
            if pos.shape[0] == 0 or neg.shape[0] == 0:
                return 0.0
            n = proj_col.shape[0]
            diff = pos.mean() - neg.mean()
            std = proj_col.std()
            if std < 1e-10:
                return 0.0
            return float(diff / std * ((pos.shape[0] * neg.shape[0] / n**2) ** 0.5))

        pbc_buggy = [_pbc(projections[:, k], labels_buggy) for k in range(min(5, args.n_components))]
        pbc_correct = [_pbc(projections[:, k], labels_correct) for k in range(min(5, args.n_components))]

        logger.info(f"  Layer {layer}: PC1 explains {explained_var_ratio[0]:.3f} of variance")
        logger.info(f"  PBC(is_buggy):    {pbc_buggy}")
        logger.info(f"  PBC(will_correct):{pbc_correct}")

        # Now compute full per-sample trajectories projected onto top-2 PCs
        traj_projections: list[dict] = []
        for meta in index:
            sid = meta["sample_id"]
            pt = traj_subdir / f"{sid}.pt"
            if not pt.exists():
                continue
            sample = torch.load(pt, map_location="cpu", weights_only=False)
            traj = sample.get("trajectory", {})
            h = traj.get(layer)
            if h is None:
                h = traj.get(str(layer))
            if h is None or h.shape[0] == 0:
                continue
            h_c = h.float() - mu
            proj = (h_c @ components[:2].T).tolist()  # [T, 2]
            traj_projections.append({
                "sample_id": sid,
                "is_buggy": meta.get("is_buggy", False),
                "correct": meta.get("correct", False),
                "mutation_type": meta.get("mutation_type", ""),
                "proj": proj,
            })

        all_layer_results[layer] = {
            "components": components.half(),  # [K, dim] fp16
            "explained_var_ratio": explained_var_ratio[: args.n_components].tolist(),
            "feature_mean": mu.squeeze(0),
            "pbc_is_buggy": pbc_buggy,
            "pbc_will_correct": pbc_correct,
            "traj_projections": traj_projections,
        }

        print(f"\nLayer {layer}: top-5 PCs explain {explained_var_ratio[:5].sum():.3f} of variance")
        print(f"  PBC(is_buggy) for PC1-5: {[f'{x:.3f}' for x in pbc_buggy]}")
        print(f"  PBC(will_correct) for PC1-5: {[f'{x:.3f}' for x in pbc_correct]}")

    out = traj_path / "pca_trajectory.pt"
    torch.save(all_layer_results, out)
    logger.info(f"Saved PCA results to {out}")


if __name__ == "__main__":
    args = load_from_cli(PCATrajectoryArgs)
    run_pca_trajectory(args)
