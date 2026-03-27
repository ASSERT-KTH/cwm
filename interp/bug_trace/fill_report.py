"""Read analysis output files and fill in the results tables in REPORT.md.

Run this after bug_analysis_cpu.sh and bug_patch.sh complete.

Usage:
    python -m interp.bug_trace.fill_report \
        traj_dir=./interp-bug-trajectories-track_a \
        report_path=experiments/02_bug_trace/REPORT.md
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from cwm.common.params import load_from_cli

logger = logging.getLogger(__name__)

LAYERS = [16, 32, 48, 63]
N_BINS = 10


@dataclass
class FillReportArgs:
    traj_dir: str = "interp-bug-trajectories-track_a"
    report_path: str = "experiments/02_bug_trace/REPORT.md"


def _fmt(x, fmt=".3f"):
    if x is None or (isinstance(x, float) and x != x):
        return "?"
    return format(x, fmt)


def _load_index(traj_path: Path) -> list[dict]:
    idx_path = traj_path / "index.jsonl"
    if not idx_path.exists():
        return []
    entries = []
    with idx_path.open() as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


def build_logit_lens_section(traj_path: Path) -> str:
    ll_path = traj_path / "logit_lens_temporal.pt"
    if not ll_path.exists():
        return "*(logit_lens_temporal.pt not found — run logit_lens_temporal first)*"

    results = torch.load(ll_path, map_location="cpu", weights_only=False)

    # Compute mean T* per layer × condition
    table: dict[int, dict[str, list]] = {L: {"orig": [], "buggy": []} for L in LAYERS}
    for r in results:
        is_buggy = r.get("is_buggy", False)
        for layer_key, ld in r.get("layers", {}).items():
            layer = int(layer_key)
            if layer not in table:
                continue
            t_star = ld.get("t_star")
            n_steps = ld.get("n_steps", 1)
            if t_star is not None and n_steps > 0:
                rel = t_star / n_steps
                key = "buggy" if is_buggy else "orig"
                table[layer][key].append(rel)

    rows = []
    for L in LAYERS:
        orig = table[L]["orig"]
        buggy = table[L]["buggy"]
        orig_mean = sum(orig) / len(orig) if orig else None
        buggy_mean = sum(buggy) / len(buggy) if buggy else None
        rows.append(
            f"| {L:<5} | {_fmt(orig_mean)} (n={len(orig)}) "
            f"| {_fmt(buggy_mean)} (n={len(buggy)}) | |"
        )

    return "\n".join([
        "| Layer | Mean T*/T (original) | Mean T*/T (buggy) | Notes |",
        "|-------|---------------------|------------------|-------|",
    ] + rows)


def build_probe_section(traj_path: Path, target: str) -> str:
    probe_path = traj_path / f"probe_temporal_{target}.pt"
    if not probe_path.exists():
        return f"*(probe_temporal_{target}.pt not found)*"

    data = torch.load(probe_path, map_location="cpu", weights_only=False)
    heatmap = data["heatmap"]

    header = "| Layer ↓ / Time bin → |" + " | ".join(f"bin{b:02d}" for b in range(N_BINS)) + " |"
    sep = "|----------------------|" + "|".join(["------"] * N_BINS) + "|"

    rows = []
    best_acc = 0.0
    best_loc = None
    for L in LAYERS:
        row_vals = []
        for b in range(N_BINS):
            val = heatmap.get(L, {}).get(b, {}).get("val_acc", float("nan"))
            row_vals.append(_fmt(val))
            if val == val and val > best_acc:
                best_acc = val
                best_loc = (L, b)
        rows.append(f"| {L:<20} | " + " | ".join(row_vals) + " |")

    best_line = f"\n**Best val_acc ({target})**: {_fmt(best_acc)} at layer {best_loc[0] if best_loc else '?'}, bin {best_loc[1] if best_loc else '?'}"

    return "\n".join([header, sep] + rows) + best_line


def build_ccs_section(traj_path: Path) -> str:
    ccs_path = traj_path / "ccs_mean.pt"
    if not ccs_path.exists():
        return "*(ccs_mean.pt not found)*"

    data = torch.load(ccs_path, map_location="cpu", weights_only=False)
    results = data.get("results", {})

    rows = []
    for L in LAYERS:
        r = results.get(L, {})
        loss = r.get("loss")
        acc = r.get("separation_acc")
        n = r.get("n_pairs")
        rows.append(
            f"| {L:<5} | {_fmt(loss)} | {_fmt(acc)} | {n or '?':<7} | |"
        )

    return "\n".join([
        "| Layer | CCS loss | Separation acc | n_pairs | Notes |",
        "|-------|----------|----------------|---------|-------|",
    ] + rows)


def build_change_point_section(traj_path: Path) -> str:
    lines = [
        "| Layer | Method | Mean T*/T (original) | Mean T*/T (buggy) | Notes |",
        "|-------|--------|----------------------|-------------------|-------|",
    ]
    for method in ["cosine", "l2"]:
        cp_path = traj_path / f"change_points_{method}.pt"
        if not cp_path.exists():
            lines.append(f"| — | {method} | *(not found)* | *(not found)* | |")
            continue
        results = torch.load(cp_path, map_location="cpu", weights_only=False)
        for L in [32]:
            orig = [r["layers"][L]["t_star_relative"]
                    for r in results
                    if L in r.get("layers", {}) and not r.get("is_buggy")]
            buggy = [r["layers"][L]["t_star_relative"]
                     for r in results
                     if L in r.get("layers", {}) and r.get("is_buggy")]
            orig_m = sum(orig) / len(orig) if orig else None
            buggy_m = sum(buggy) / len(buggy) if buggy else None
            lines.append(
                f"| {L:<5} | {method:<6} | {_fmt(orig_m)} (n={len(orig)}) "
                f"| {_fmt(buggy_m)} (n={len(buggy)}) | |"
            )
    return "\n".join(lines)


def build_dmd_section(traj_path: Path) -> str:
    dmd_path = traj_path / "dmd.pt"
    if not dmd_path.exists():
        return "*(dmd.pt not found)*"

    data = torch.load(dmd_path, map_location="cpu", weights_only=False)
    results = data.get("results", {})

    rows = []
    for L in LAYERS:
        layer_data = results.get(L, {})
        dmd_all = layer_data.get("dmd_all", {})
        mags = dmd_all.get("magnitudes")
        if mags is None:
            rows.append(f"| {L:<5} | ? | ? | |")
        else:
            n_persist = int((mags > 0.95).sum().item())
            top_mag = float(mags.max().item())
            rows.append(f"| {L:<5} | {n_persist:<29} | {_fmt(top_mag)} | |")

    return "\n".join([
        "| Layer | n_persistent modes (|λ|>0.95) | Top eigenvalue magnitude | Notes |",
        "|-------|-------------------------------|--------------------------|-------|",
    ] + rows)


def build_pca_section(traj_path: Path) -> str:
    pca_path = traj_path / "pca_trajectory.pt"
    if not pca_path.exists():
        return "*(pca_trajectory.pt not found)*"

    data = torch.load(pca_path, map_location="cpu", weights_only=False)

    lines = []
    for L in LAYERS:
        layer_data = data.get(L)
        if layer_data is None:
            continue
        evr = layer_data.get("explained_var_ratio", [])
        pbc_buggy = layer_data.get("pbc_is_buggy", [])
        pbc_correct = layer_data.get("pbc_will_correct", [])
        top5_var = sum(evr[:5]) if evr else 0.0
        lines.append(f"\n**Layer {L}**:")
        lines.append(f"- Top-5 PCs explain {_fmt(top5_var)} of variance")
        lines.append(f"- PBC(is_buggy) for PC1-5: {[_fmt(x) for x in pbc_buggy[:5]]}")
        lines.append(f"- PBC(will_correct) for PC1-5: {[_fmt(x) for x in pbc_correct[:5]]}")

    return "\n".join(lines) if lines else "*(no PCA results)*"


def fill_report(args: FillReportArgs) -> None:
    logging.basicConfig(level=logging.INFO)
    traj_path = Path(args.traj_dir)
    report_path = Path(args.report_path)

    if not report_path.exists():
        logger.error(f"Report not found: {report_path}")
        return

    with report_path.open() as f:
        content = f.read()

    replacements = {
        "| Layer | Mean T* (original) | Mean T* (buggy) | Notes |\n|-------|--------------------|-----------------|-------|\n| 16    | ?                  | ?               | |\n| 32    | ?                  | ?               | |\n| 48    | ?                  | ?               | |\n| 63    | ?                  | ?               | |":
            build_logit_lens_section(traj_path),

        "| Layer | CCS loss | Separation acc | n_pairs | Notes |\n|-------|----------|----------------|---------|-------|\n| 16    | ?        | ?              | ?       | |\n| 32    | ?        | ?              | ?       | |\n| 48    | ?        | ?              | ?       | |\n| 63    | ?        | ?              | ?       | |":
            build_ccs_section(traj_path),

        "| Layer | Method | Mean T*/T (original) | Mean T*/T (buggy) | Notes |\n|-------|--------|----------------------|-------------------|-------|\n| 32    | cosine | ?                    | ?                 | |\n| 32    | l2     | ?                    | ?                 | |":
            build_change_point_section(traj_path),

        "| Layer | n_persistent modes (|λ|>0.95) | Top eigenvalue magnitude | Notes |\n|-------|-------------------------------|--------------------------|-------|\n| 32    | ?                             | ?                        | |\n| 48    | ?                             | ?                        | |":
            build_dmd_section(traj_path),
    }

    for placeholder, replacement in replacements.items():
        if placeholder in content:
            content = content.replace(placeholder, replacement)
            logger.info(f"Replaced a table placeholder")
        else:
            logger.warning(f"Placeholder not found (table may already be filled)")

    # Handle probe tables (two targets)
    for target in ["is_buggy", "will_be_correct"]:
        probe_placeholder = (
            f"**Results — {target}**: *(fill in)*\n\n"
            "| Layer ↓ / Time bin → | bin0 | bin1 | bin2 | ... | bin9 |\n"
            "|----------------------|------|------|------|-----|------|\n"
            "| 16                   | ?    | ?    | ?    | ... | ?    |\n"
            "| 32                   | ?    | ?    | ?    | ... | ?    |\n"
            "| 48                   | ?    | ?    | ?    | ... | ?    |\n"
            "| 63                   | ?    | ?    | ?    | ... | ?    |\n"
            "\n**Best val_acc (is_buggy)**: ? at layer ?, bin ?\n"
            "**Best val_acc (will_be_correct)**: ? at layer ?, bin ?"
        )
        probe_content = f"**Results — {target}**:\n\n{build_probe_section(traj_path, target)}"
        if "| 16                   | ?    | ?    | ?    | ... | ?    |" in content:
            # Replace both probe tables together once
            combined_placeholder = (
                "**Results — is_buggy**: *(fill in)*\n\n"
                "| Layer ↓ / Time bin → | bin0 | bin1 | bin2 | ... | bin9 |\n"
                "|----------------------|------|------|------|-----|------|\n"
                "| 16                   | ?    | ?    | ?    | ... | ?    |\n"
                "| 32                   | ?    | ?    | ?    | ... | ?    |\n"
                "| 48                   | ?    | ?    | ?    | ... | ?    |\n"
                "| 63                   | ?    | ?    | ?    | ... | ?    |\n"
                "\n**Best val_acc (is_buggy)**: ? at layer ?, bin ?\n"
                "**Best val_acc (will_be_correct)**: ? at layer ?, bin ?"
            )
            combined_replacement = (
                f"**Results — is_buggy**:\n\n{build_probe_section(traj_path, 'is_buggy')}\n\n"
                f"**Results — will_be_correct**:\n\n{build_probe_section(traj_path, 'will_be_correct')}"
            )
            content = content.replace(combined_placeholder, combined_replacement, 1)
            break

    # PCA section
    pca_placeholder = "*(fill in: explained variance ratio, PBC of PC1 for is_buggy and will_be_correct)*"
    if pca_placeholder in content:
        content = content.replace(pca_placeholder, build_pca_section(traj_path))

    # Update status line
    content = content.replace(
        "> **Status**: Scaffold — fill in each section as experiments complete.",
        "> **Status**: Results filled in from analysis outputs.",
    )
    content = content.replace(
        "> Last updated: *auto-updated by analysis scripts*",
        f"> Last updated: auto-filled by `fill_report.py` from `{args.traj_dir}`",
    )

    with report_path.open("w") as f:
        f.write(content)

    logger.info(f"Updated {report_path}")
    print(f"\nReport updated: {report_path}")
    print(f"Results sections filled from: {traj_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = load_from_cli(FillReportArgs)
    fill_report(args)
