# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Table 9 metric computation: a component-wise breakdown of trace-prediction
quality. Given the ground-truth frames (from ``ground_truth.py``) and the
parsed model frames (from ``trace_format.py``), we compute, per trace:

  * Valid Trace Format      - did the whole generation parse cleanly?
  * State Exact Match       - fraction of GT observation states reproduced exactly
  * Action Exact Match      - fraction of GT actions (source lines) reproduced exactly
  * Valid JSON Format       - fraction of predicted states that are valid JSON objects
  * Key Match               - mean per-state Jaccard overlap of variable names
  * Key+Value Match         - mean per-state Jaccard overlap of (name, value) pairs

plus dataset statistics (avg state / action length in tokens, computed over the
ground-truth traces). Frames are aligned positionally, which is exact when the
prediction is well-formed and degrades gracefully (truncating to the shorter
trace) when it is not.

The reported table value for each rate is the macro-average over traces (mean
of per-trace fractions), matching the paper's "... per execution trace"
phrasing. Output pass@1 is computed separately in ``run_eval`` (it reuses the
CRUXEval answer extractor and executes the assertion).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from dataset.cruxeval.trace_format import TraceEvent, TraceFrame


@dataclass
class TraceMetrics:
    valid_trace_format: float = 0.0
    # state/action exact match are None when the GT trace has no states/actions
    state_exact_match: float | None = None
    action_exact_match: float | None = None
    valid_json_format: float | None = None
    key_match: float | None = None
    key_value_match: float | None = None
    avg_state_tokens: float | None = None
    avg_action_tokens: float | None = None


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def compute_trace_metrics(
    gt_frames: list[TraceFrame],
    pred_frames: list[TraceFrame],
    well_formed: bool,
    token_len: Callable[[str], int] | None = None,
) -> TraceMetrics:
    """Compute the per-trace Table 9 components for one sample."""
    m = TraceMetrics(valid_trace_format=1.0 if well_formed else 0.0)

    gt_obs = [f for f in gt_frames if f.has_locals]
    n_pred = len(pred_frames)

    # --- Action exact match (over all GT frames) ---
    if gt_frames:
        action_hits = 0
        for i, gf in enumerate(gt_frames):
            if i < n_pred and pred_frames[i].source == gf.source:
                action_hits += 1
        m.action_exact_match = action_hits / len(gt_frames)

    # --- State exact / Valid JSON / Key / Key+Value (over GT observation frames) ---
    if gt_obs:
        # Map GT observation frames back to their global index for alignment.
        obs_indices = [i for i, f in enumerate(gt_frames) if f.has_locals]
        state_hits = 0
        json_valid = 0
        key_scores: list[float] = []
        kv_scores: list[float] = []
        for gi in obs_indices:
            gf = gt_frames[gi]
            pf = pred_frames[gi] if gi < n_pred else None
            pred_locals = (
                pf.locals if (pf is not None and pf.has_locals) else None
            )
            if pf is not None and pf.has_locals and pf.locals is not None:
                json_valid += 1
            if pred_locals is not None and gf.locals is not None:
                if pred_locals == gf.locals:
                    state_hits += 1
                gk, pk = set(gf.locals), set(pred_locals)
                key_scores.append(_jaccard(gk, pk))
                gkv = set(gf.locals.items())
                pkv = set(pred_locals.items())
                kv_scores.append(_jaccard(gkv, pkv))
            else:
                key_scores.append(0.0)
                kv_scores.append(0.0)
        m.state_exact_match = state_hits / len(gt_obs)
        m.valid_json_format = json_valid / len(gt_obs)
        m.key_match = sum(key_scores) / len(key_scores)
        m.key_value_match = sum(kv_scores) / len(kv_scores)

    # --- Dataset statistics over the ground-truth trace ---
    if token_len is not None:
        import json as _json

        if gt_obs:
            m.avg_state_tokens = sum(
                token_len(_json.dumps(f.locals)) for f in gt_obs
            ) / len(gt_obs)
        if gt_frames:
            m.avg_action_tokens = sum(
                token_len(f.source) for f in gt_frames
            ) / len(gt_frames)

    return m


@dataclass
class Table9Aggregate:
    n: int = 0
    output_pass_at_1: float = 0.0
    valid_trace_format: float = 0.0
    state_exact_match: float = 0.0
    action_exact_match: float = 0.0
    valid_json_format: float = 0.0
    key_match: float = 0.0
    key_value_match: float = 0.0
    avg_state_tokens: float = 0.0
    avg_action_tokens: float = 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate(
    per_sample: list[TraceMetrics], pass_at_1: list[float]
) -> Table9Aggregate:
    """Macro-average per-trace metrics into a single Table 9 column."""

    def col(attr: str) -> list[float]:
        return [
            getattr(m, attr) for m in per_sample if getattr(m, attr) is not None
        ]

    agg = Table9Aggregate(
        n=len(per_sample),
        output_pass_at_1=_mean(pass_at_1) * 100,
        valid_trace_format=_mean([m.valid_trace_format for m in per_sample]) * 100,
        state_exact_match=_mean(col("state_exact_match")) * 100,
        action_exact_match=_mean(col("action_exact_match")) * 100,
        valid_json_format=_mean(col("valid_json_format")) * 100,
        key_match=_mean(col("key_match")) * 100,
        key_value_match=_mean(col("key_value_match")) * 100,
        avg_state_tokens=_mean(col("avg_state_tokens")),
        avg_action_tokens=_mean(col("avg_action_tokens")),
    )
    return agg


def format_table9(agg: Table9Aggregate, column_label: str = "CruxEval") -> str:
    """Render a Table 9 style summary for a single dataset column."""
    rows = [
        ("Output", "pass@1", agg.output_pass_at_1),
        ("Trace", "Valid Trace Format", agg.valid_trace_format),
        ("", "State Exact Match", agg.state_exact_match),
        ("", "Action Exact Match", agg.action_exact_match),
        ("States", "Valid JSON Format", agg.valid_json_format),
        ("", "Key Match", agg.key_match),
        ("", "Key+Value Match", agg.key_value_match),
        ("Statistics", "Avg State Length (Token)", agg.avg_state_tokens),
        ("", "Avg Action Length (Token)", agg.avg_action_tokens),
    ]
    width = max(len(label) for _, label, _ in rows)
    lines = [f"{'':<12}{'':<{width}}  {column_label:>8}", "-" * (12 + width + 12)]
    for group, label, val in rows:
        lines.append(f"{group:<12}{label:<{width}}  {val:>8.1f}")
    lines.append(f"\n(n={agg.n} samples)")
    return "\n".join(lines)
