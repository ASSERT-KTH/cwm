"""Dynamic Mode Decomposition (DMD) on decode-step trajectories.

Fits a linear dynamical system h_{T+1} ≈ A · h_T to the trajectory data.
The eigenvalues of A reveal dominant temporal modes:
  - |λ| ≈ 1: persistent/oscillatory mode (candidate "working memory")
  - |λ| < 1: decaying mode (transient information)
  - |λ| > 1: growing mode (rare, indicates non-stationarity)

For interpretability, we project A onto a low-dim subspace via PCA first
(standard "Exact DMD" in reduced coordinates).

Usage:
    python -m interp.bug_trace.analysis.dmd \\
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
class DMDArgs:
    traj_dir: str = "interp-bug-trajectories"
    layers: list[int] = field(default_factory=lambda: [16, 32, 48, 63])
    n_dmd_modes: int = 16       # rank of DMD approximation
    min_traj_len: int = 10      # minimum trajectory length to include
    max_trajs_per_condition: int = 150  # subsample for tractable SVD (~150*160 pairs)
    seed: int = 42


def _exact_dmd(
    trajectories: list[torch.Tensor],  # list of [T, dim] tensors
    r: int,                             # number of DMD modes
) -> dict:
    """Compute Exact DMD from a collection of trajectories.

    Stacks all (h_T, h_{T+1}) pairs, fits A = X' X^+ via SVD.
    Returns eigenvalues, eigenvectors, and mode participation scores.
    """
    X_list, Xp_list = [], []
    for h in trajectories:
        if h.shape[0] < 2:
            continue
        X_list.append(h[:-1].float())    # [T-1, dim]
        Xp_list.append(h[1:].float())    # [T-1, dim]

    if not X_list:
        return {}

    X = torch.cat(X_list, dim=0)   # [N_pairs, dim]
    Xp = torch.cat(Xp_list, dim=0)

    # Randomised truncated SVD — only computes top-r modes (much faster than full SVD)
    U, S, V = torch.svd_lowrank(X, q=r, niter=4)
    Vh = V.T  # [r, dim]

    # Reduced DMD operator: A_tilde = U^T X' V S^{-1}
    S_inv = 1.0 / S.clamp(min=1e-8)
    A_tilde = (U.T @ Xp) @ Vh.T @ torch.diag(S_inv)  # [r, r]

    # Eigendecomposition of A_tilde
    eigvals, eigvecs = torch.linalg.eig(A_tilde.to(torch.complex64))
    # eigvecs: [r, r] (columns are eigenvectors)

    # DMD modes in original space: Phi = X' V S^{-1} W
    # W = eigvecs (complex); cast real matrices to complex for multiplication
    Xp_c = Xp.to(torch.complex64)
    Vh_c = Vh.to(torch.complex64)
    S_inv_diag = torch.diag(S_inv.to(torch.complex64))
    Phi = Xp_c @ Vh_c.T @ S_inv_diag @ eigvecs  # [N_pairs, r]

    # Sort by eigenvalue magnitude (most persistent first)
    magnitudes = eigvals.abs()
    sort_idx = magnitudes.argsort(descending=True)
    eigvals = eigvals[sort_idx]
    eigvecs = eigvecs[:, sort_idx]
    Phi = Phi[:, sort_idx]
    magnitudes = magnitudes[sort_idx]

    return {
        "eigenvalues": eigvals.cpu(),           # [r] complex
        "magnitudes": magnitudes.cpu(),          # [r] real
        "eigenvectors": eigvecs.cpu(),           # [r, r] complex (in reduced space)
        "modes": Phi.cpu(),                      # [N_pairs, r] complex (in full space)
        "A_tilde": A_tilde.cpu(),               # [r, r]
        "U": U.cpu(),                            # [N_pairs, r]
        "S": S.cpu(),                            # [r]
        "Vh": Vh.cpu(),                          # [r, dim]
    }


def run_dmd(args: DMDArgs) -> None:
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(args.seed)

    traj_path = Path(args.traj_dir)
    index: list[dict] = []
    with (traj_path / "index.jsonl").open() as f:
        for line in f:
            index.append(json.loads(line))
    traj_subdir = traj_path / "trajectories"

    all_results: dict = {}

    # Process one layer at a time to avoid OOM (loading all layers simultaneously
    # at float32 with 1280 samples × ~256 steps × 6144 dims × 4 layers ≈ 7.5 GB).
    for layer in args.layers:
        trajs_orig: list[torch.Tensor] = []
        trajs_buggy: list[torch.Tensor] = []

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
            if h is None or h.shape[0] < args.min_traj_len:
                continue
            if meta.get("is_buggy", False):
                trajs_buggy.append(h.float())
            else:
                trajs_orig.append(h.float())

        # Subsample to keep SVD tractable on CPU
        rng = torch.Generator().manual_seed(args.seed + layer)
        if len(trajs_orig) > args.max_trajs_per_condition:
            idx = torch.randperm(len(trajs_orig), generator=rng)[:args.max_trajs_per_condition]
            trajs_orig = [trajs_orig[i] for i in idx.tolist()]
        if len(trajs_buggy) > args.max_trajs_per_condition:
            idx = torch.randperm(len(trajs_buggy), generator=rng)[:args.max_trajs_per_condition]
            trajs_buggy = [trajs_buggy[i] for i in idx.tolist()]

        logger.info(
            f"Layer {layer}: DMD on {len(trajs_orig)} original + {len(trajs_buggy)} buggy"
        )

        r = min(args.n_dmd_modes, 64)  # cap at 64 modes

        dmd_orig = _exact_dmd(trajs_orig, r) if trajs_orig else {}
        dmd_buggy = _exact_dmd(trajs_buggy, r) if trajs_buggy else {}
        dmd_all = _exact_dmd(trajs_orig + trajs_buggy, r)

        all_results[layer] = {
            "dmd_original": dmd_orig,
            "dmd_buggy": dmd_buggy,
            "dmd_all": dmd_all,
        }

        if dmd_all.get("magnitudes") is not None:
            mags = dmd_all["magnitudes"]
            print(f"\nLayer {layer}: top-5 DMD eigenvalue magnitudes: {mags[:5].tolist()}")
            n_persistent = int((mags > 0.95).sum().item())
            print(f"  {n_persistent}/{r} modes with |λ| > 0.95 (persistent state)")

        if dmd_orig.get("magnitudes") is not None and dmd_buggy.get("magnitudes") is not None:
            mags_orig = dmd_orig["magnitudes"][:5].tolist()
            mags_buggy = dmd_buggy["magnitudes"][:5].tolist()
            print(f"  Original top-5 mag: {[f'{m:.3f}' for m in mags_orig]}")
            print(f"  Buggy    top-5 mag: {[f'{m:.3f}' for m in mags_buggy]}")

        # Explicitly free trajectory tensors before next layer pass
        del trajs_orig, trajs_buggy

    out = traj_path / "dmd.pt"
    torch.save({"results": all_results, "args": vars(args)}, out)
    logger.info(f"Saved DMD results to {out}")


if __name__ == "__main__":
    args = load_from_cli(DMDArgs)
    run_dmd(args)
