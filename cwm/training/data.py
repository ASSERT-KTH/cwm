# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Ground-truth execution traces -> (input_ids, labels) for teacher-forcing CODI.

Thin HuggingFace-tokenizer wrapper over the verbatim Table 9 trace generator
(``evals.trace_analysis``). All trace logic is reused: we only build the seeded
prompt, tokenize ``prompt + render_frames_to_generation(frames)``, and mask the
prompt out of the labels (the student is teacher-forced, so labels == input_ids
with the prompt prefix set to ``-100``).
"""

from __future__ import annotations

from evals.trace_analysis.ground_truth import ground_truth_trace, make_trace_context
from evals.trace_analysis.trace_format import render_frames_to_generation

IGNORE_INDEX = -100


def _prompt_str(code: str, input_str: str) -> str:
    ctx = make_trace_context(code, input_str)
    return f"<|trace_context_start|>{ctx}<|frame_sep|><|call_sep|>{{}}<|action_sep|>def main():\n<|frame_sep|>"


def build_example(
    code: str, input_str: str, tokenizer, *, max_seq_len: int
) -> tuple[list[int], list[int]] | None:
    """Return ``(input_ids, labels)``, or None to skip (empty / too long).

    A raised program is kept: its EXCEPTION frame is part of the trace to predict.
    ``render_frames_to_generation`` already terminates the trace with ``<|end_of_text|>``.
    """
    frames, _error = ground_truth_trace(code, input_str, align_to_prompt=True)
    if not frames:
        return None
    prompt_ids = [tokenizer.bos_token_id] + tokenizer.encode(
        _prompt_str(code, input_str), add_special_tokens=False
    )
    trace_ids = tokenizer.encode(render_frames_to_generation(frames), add_special_tokens=False)
    input_ids = prompt_ids + trace_ids
    if len(input_ids) > max_seq_len:
        return None
    return input_ids, [IGNORE_INDEX] * len(prompt_ids) + trace_ids


def build_dataset(tokenizer, *, n_samples: int = -1, max_seq_len: int = 8192) -> list[tuple[list[int], list[int]]]:
    """Tokenized CRUXEval-O traces. ``n_samples<=0`` uses all 800."""
    from datasets import load_dataset

    rows = list(load_dataset("cruxeval-org/cruxeval", split="test"))
    if n_samples > 0:
        rows = rows[:n_samples]
    examples = (build_example(r["code"], r["input"], tokenizer, max_seq_len=max_seq_len) for r in rows)
    return [ex for ex in examples if ex is not None]
