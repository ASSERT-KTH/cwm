"""Autonomous decision logic: reads analysis results and recommends next experiments.

Reads results from traj_dir and prints a machine-readable recommendation for
what to run next in the iteration loop.

Exit codes:
  0: proceed to next tier
  1: insufficient data, re-run extraction
  2: all tiers complete, synthesis done

Usage:
    python -m interp.bug_trace.decide_next \\
        traj_dir=./interp-bug-trajectories \\
        current_tier=1
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from cwm.common.params import load_from_cli

logger = logging.getLogger(__name__)

# Thresholds for "interesting" findings
PROBE_ACC_THRESHOLD = 0.60      # probe accuracy above chance
CCS_ACC_THRESHOLD = 0.60        # CCS separation accuracy
STEERING_DELTA_THRESHOLD = 0.05 # pass@1 improvement from steering


@dataclass
class DecideNextArgs:
    traj_dir: str = "interp-bug-trajectories"
    current_tier: int = 1


def _check_index(traj_path: Path) -> dict:
    idx_path = traj_path / "index.jsonl"
    if not idx_path.exists():
        return {"ok": False, "msg": "No index.jsonl found"}
    entries = []
    with idx_path.open() as f:
        for line in f:
            entries.append(json.loads(line))
    n_buggy = sum(e.get("is_buggy", False) for e in entries)
    n_orig = len(entries) - n_buggy
    return {"ok": True, "n_total": len(entries), "n_buggy": n_buggy, "n_orig": n_orig}


def _check_tier1(traj_path: Path) -> dict:
    """Check Tier 1 results (logit lens + probe)."""
    findings = []
    recommendations = []
    proceed = True

    # Logit lens temporal
    ll_path = traj_path / "logit_lens_temporal.pt"
    if ll_path.exists():
        results = torch.load(ll_path, map_location="cpu", weights_only=False)
        # Check: does any layer show T* < 50% for original traces?
        early_commit = False
        for r in results:
            for layer_data in r.get("layers", {}).values():
                t_star_rel = None
                t_star = layer_data.get("t_star")
                n_steps = layer_data.get("n_steps", 1)
                if t_star is not None and n_steps > 0:
                    t_star_rel = t_star / n_steps
                    if t_star_rel < 0.5 and not r.get("is_buggy"):
                        early_commit = True
        if early_commit:
            findings.append("LOGIT_LENS: early commitment (T* < 50%) found in original traces")
        else:
            findings.append("LOGIT_LENS: no strong early commitment found")
    else:
        recommendations.append("RUN: logit_lens_temporal")
        proceed = False

    # Probe temporal (is_buggy)
    probe_path = traj_path / "probe_temporal_is_buggy.pt"
    if probe_path.exists():
        data = torch.load(probe_path, map_location="cpu", weights_only=False)
        heatmap = data["heatmap"]
        best_acc = 0.0
        best_loc = None
        for layer, bins in heatmap.items():
            for b, bdata in bins.items():
                acc = bdata.get("val_acc", 0)
                if acc > best_acc:
                    best_acc = acc
                    best_loc = (layer, b)
        findings.append(f"PROBE_IS_BUGGY: best val_acc={best_acc:.3f} at {best_loc}")
        if best_acc >= PROBE_ACC_THRESHOLD:
            findings.append("→ REPRESENTATION EXISTS (probe > threshold)")
        else:
            findings.append("→ WEAK REPRESENTATION (probe below threshold)")
            recommendations.append("CONSIDER: try Track B (trace_full) instead")
    else:
        recommendations.append("RUN: probe_temporal is_buggy")
        proceed = False

    probe_path2 = traj_path / "probe_temporal_will_be_correct.pt"
    if probe_path2.exists():
        data = torch.load(probe_path2, map_location="cpu", weights_only=False)
        heatmap = data["heatmap"]
        best_acc = max(
            (bdata.get("val_acc", 0) for bins in heatmap.values() for bdata in bins.values()),
            default=0.0,
        )
        findings.append(f"PROBE_WILL_CORRECT: best val_acc={best_acc:.3f}")
    else:
        recommendations.append("RUN: probe_temporal will_be_correct")

    return {"findings": findings, "recommendations": recommendations, "proceed": proceed}


def _check_tier2(traj_path: Path) -> dict:
    findings = []
    recommendations = []
    proceed = True

    # CCS
    ccs_path = traj_path / "ccs_mean.pt"
    if ccs_path.exists():
        data = torch.load(ccs_path, map_location="cpu", weights_only=False)
        results = data.get("results", {})
        best_acc = max((r.get("separation_acc", 0) for r in results.values()), default=0)
        findings.append(f"CCS: best separation_acc={best_acc:.3f}")
        if best_acc >= CCS_ACC_THRESHOLD:
            findings.append("→ CCS DIRECTION FOUND (strong label-free separation)")
        else:
            findings.append("→ CCS DIRECTION WEAK")
    else:
        recommendations.append("RUN: ccs time_bin=mean")
        proceed = False

    # Change point
    for method in ["cosine", "l2"]:
        cp_path = traj_path / f"change_points_{method}.pt"
        if cp_path.exists():
            results = torch.load(cp_path, map_location="cpu", weights_only=False)
            t_rels = [
                r["layers"][32]["t_star_relative"]
                for r in results
                if 32 in r.get("layers", {}) and not r.get("is_buggy")
            ]
            if t_rels:
                mean_rel = sum(t_rels) / len(t_rels)
                findings.append(f"CHANGE_POINT_{method}: mean T*/T = {mean_rel:.3f}")
                if mean_rel < 0.3:
                    findings.append("→ EARLY AHA MOMENT (T* in first 30% of generation)")
                elif mean_rel > 0.7:
                    findings.append("→ LATE AHA MOMENT (T* in last 30% of generation)")
                else:
                    findings.append("→ MID-GENERATION AHA MOMENT")
        else:
            recommendations.append(f"RUN: change_point method={method}")

    return {"findings": findings, "recommendations": recommendations, "proceed": proceed}


def _check_tier3(traj_path: Path) -> dict:
    findings = []
    recommendations = []

    # RSA
    for suffix in ["mean", "last_quarter"]:
        rsa_path = traj_path / f"rsa_{suffix}.pt"
        if rsa_path.exists():
            data = torch.load(rsa_path, map_location="cpu", weights_only=False)
            results = data.get("results", {})
            for layer, r in results.items():
                rho = r.get("spearman_edit_dist", 0)
                findings.append(f"RSA_{suffix}: layer {layer} ρ(edit_dist)={rho:.4f}")

    # DMD
    dmd_path = traj_path / "dmd.pt"
    if dmd_path.exists():
        data = torch.load(dmd_path, map_location="cpu", weights_only=False)
        for layer, r in data.get("results", {}).items():
            dmd_all = r.get("dmd_all", {})
            mags = dmd_all.get("magnitudes")
            if mags is not None:
                n_persist = int((mags > 0.95).sum().item())
                findings.append(f"DMD: layer {layer} has {n_persist} persistent modes (|λ|>0.95)")

    # Patching
    patch_path = traj_path / "patch_results.jsonl"
    if (Path(traj_path).parent / "interp-bug-patch" / "patch_results.jsonl").exists():
        findings.append("PATCHING: results available — check patch_results.jsonl")

    return {"findings": findings, "recommendations": recommendations}


def run_decide_next(args: DecideNextArgs) -> None:
    logging.basicConfig(level=logging.INFO)
    traj_path = Path(args.traj_dir)

    print("=" * 60)
    print(f"DECISION ENGINE — traj_dir={args.traj_dir}, tier={args.current_tier}")
    print("=" * 60)

    # Check data availability
    idx = _check_index(traj_path)
    if not idx["ok"]:
        print("STATUS: NO_DATA")
        print("ACTION: RUN_EXTRACTION")
        sys.exit(1)
    print(f"DATA: {idx['n_total']} samples ({idx['n_orig']} orig, {idx['n_buggy']} buggy)")

    all_findings = []
    all_recommendations = []

    if args.current_tier >= 1:
        t1 = _check_tier1(traj_path)
        all_findings.extend(t1["findings"])
        all_recommendations.extend(t1["recommendations"])

    if args.current_tier >= 2:
        t2 = _check_tier2(traj_path)
        all_findings.extend(t2["findings"])
        all_recommendations.extend(t2["recommendations"])

    if args.current_tier >= 3:
        t3 = _check_tier3(traj_path)
        all_findings.extend(t3["findings"])
        all_recommendations.extend(t3["recommendations"])

    print("\nFINDINGS:")
    for f in all_findings:
        print(f"  {f}")

    if all_recommendations:
        print("\nRECOMMENDATIONS:")
        for r in all_recommendations:
            print(f"  {r}")

    # Final verdict
    print("\n" + "=" * 60)
    if not all_recommendations:
        if args.current_tier < 3:
            print(f"STATUS: TIER_{args.current_tier}_COMPLETE → PROCEED_TO_TIER_{args.current_tier + 1}")
            print(f"NEXT_TIER: {args.current_tier + 1}")
        else:
            print("STATUS: ALL_TIERS_COMPLETE → RUN_SYNTHESIS")
            print("ACTION: RUN_VISUALIZE AND UPDATE_REPORT")
    else:
        print(f"STATUS: TIER_{args.current_tier}_INCOMPLETE")
        print(f"NEXT_TIER: {args.current_tier}")

    # Machine-readable output
    output = {
        "tier": args.current_tier,
        "n_samples": idx.get("n_total", 0),
        "findings": all_findings,
        "recommendations": all_recommendations,
        "status": "incomplete" if all_recommendations else "complete",
    }
    with (traj_path / "decide_next.json").open("w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    args = load_from_cli(DecideNextArgs)
    run_decide_next(args)
