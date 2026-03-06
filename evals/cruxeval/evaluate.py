# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Answer extraction and execution-based scoring for CRUXEval-O.
Mirrors the evaluation logic from the original CRUXEval benchmark:
https://github.com/facebookresearch/cruxeval/blob/main/evaluation/utils_general.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys

_ARG_SEP = "<|arg_sep|>"
_FRAME_SEP = "<|frame_sep|>"
_RETURN_SEP = "<|return_sep|>"


def extract_answer(generation: str, input: str) -> str | None:
    """
    Extract the predicted output value from a model generation.

    The model is expected to generate text of the form:
        assert f(<input>) == <value>
        [/ANSWER]

    We extract the value from the RHS of ==. The original CRUXEval benchmark
    filters generations containing f({input}), but that filter is for models
    that output bare values — it would incorrectly reject every CWM generation
    since full assertions always contain the input call.
    """
    # Strip [ANSWER]/[/ANSWER] wrappers if present
    text = generation
    if "[ANSWER]" in text:
        text = text.split("[ANSWER]", 1)[-1]
    if "[/ANSWER]" in text:
        text = text.split("[/ANSWER]", 1)[0]
    text = text.strip()

    # Extract value from the RHS of == in an assert statement
    match = re.search(r"==\s*(.+?)(?:\n|$)", text)
    if match:
        return match.group(1).strip()

    return None


def extract_answer_reasoning(generation: str, input: str) -> str | None:
    """Extract answer from a reasoning-mode generation (strips <think>...</think> blocks)."""
    text = re.sub(r"<think>.*?</think>", "", generation, flags=re.DOTALL)
    return extract_answer(text, input)


def _parse_trace_value(value_str: str) -> str | None:
    """Parse the JSON-encoded return/exception value from a trace frame."""
    value_str = value_str.strip()
    if not value_str:
        return None
    try:
        return json.loads(value_str)
    except json.JSONDecodeError:
        return value_str


def extract_answer_trace_single_step(generation: str, input: str) -> str | None:
    """
    Extract predicted output from a single-step trace generation.

    The model generates: [ACTION_SEP] return f(...)[ARG_SEP]"value"[FRAME_SEP]
    We extract the JSON-encoded value between [ARG_SEP] and [FRAME_SEP].
    """
    arg_start = generation.find(_ARG_SEP)
    if arg_start == -1:
        return None
    after_arg = generation[arg_start + len(_ARG_SEP):]
    frame_end = after_arg.find(_FRAME_SEP)
    value_str = after_arg[:frame_end] if frame_end != -1 else after_arg
    return _parse_trace_value(value_str)


def extract_answer_trace_full(generation: str, input: str) -> str | None:
    """
    Extract predicted output from a full-trace generation.

    The last [RETURN_SEP] frame in the trace corresponds to main()'s return.
    Its format is: [RETURN_SEP][ACTION_SEP] return f(...)[ARG_SEP]"value"[FRAME_SEP]
    We extract the JSON-encoded value from [ARG_SEP] to [FRAME_SEP].
    Returns None if main() raised an exception instead of returning.
    """
    last_return = generation.rfind(_RETURN_SEP)
    if last_return == -1:
        return None
    return extract_answer_trace_single_step(generation[last_return:], input)


def check_correct(
    code: str,
    expected_output: str,
    predicted: str,
    timeout: float = 3.0,
) -> bool:
    """
    Check correctness by executing: {code}\\nassert {expected_output} == {predicted}

    This matches the original CRUXEval evaluation: rather than string-comparing
    the predicted value to the expected output, we run the assertion so that
    equivalent representations (e.g. 1 vs 1.0, or tuple vs list) are handled
    correctly by Python's equality semantics.
    """
    test_code = f"{code}\nassert {expected_output} == {predicted}"
    try:
        result = subprocess.run(
            [sys.executable, "-c", test_code],
            timeout=timeout,
            capture_output=True,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
