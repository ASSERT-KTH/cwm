# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Answer extraction and execution-based scoring for CRUXEval-O.
Mirrors the evaluation logic from the original CRUXEval benchmark:
https://github.com/facebookresearch/cruxeval/blob/main/evaluation/utils_general.py
"""

from __future__ import annotations

import re
import subprocess
import sys


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
