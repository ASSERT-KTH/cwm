# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Offline re-scoring of a Table 9 run (no GPU required).

Reads a ``results.jsonl`` produced by ``run_eval`` (each line must contain
``code``, ``input``, ``generation``, and ``correct``), re-parses the stored
generations, rebuilds the ground-truth traces, recomputes the metrics, and
prints the Table 9 summary. Useful for iterating on the ground-truth tracer or
metric definitions without re-running the model.

Token-length statistics require the CWM tokenizer; pass ``--tokenizer`` to
include them, otherwise the Avg *-Length rows are omitted.

    python -m evals.trace_analysis.score ./eval-cwm-table9/results.jsonl \\
        --tokenizer /path/to/cwm/tokenizer.model
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable

from evals.trace_analysis.ground_truth import ground_truth_trace
from evals.trace_analysis.metrics import (
    aggregate,
    compute_trace_metrics,
    format_table9,
)
from evals.trace_analysis.trace_format import parse_generated_trace


def _build_token_len(tokenizer_path: str | None) -> Callable[[str], int] | None:
    if tokenizer_path is None:
        return None
    from cwm.text.tokenizers import build_tokenizer

    tok = build_tokenizer("cwm_instruct", tokenizer_path)
    return lambda s: len(tok.encode(s, bos=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="Path to results.jsonl from run_eval")
    ap.add_argument("--tokenizer", default=None, help="Path to CWM tokenizer for token stats")
    ap.add_argument("--label", default="CruxEval", help="Column label")
    args = ap.parse_args()

    token_len = _build_token_len(args.tokenizer)

    per_sample = []
    pass_at_1 = []
    with open(args.results) as f:
        for line in f:
            r = json.loads(line)
            pred_frames, well_formed = parse_generated_trace(r["generation"])
            gt_frames, _ = ground_truth_trace(r["code"], r["input"])
            per_sample.append(
                compute_trace_metrics(
                    gt_frames, pred_frames, well_formed, token_len=token_len
                )
            )
            pass_at_1.append(1.0 if r.get("correct") else 0.0)

    agg = aggregate(per_sample, pass_at_1)
    print("\n" + format_table9(agg, column_label=args.label) + "\n")


if __name__ == "__main__":
    main()
