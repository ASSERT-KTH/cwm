# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Unit tests for CRUXEval-O evaluation modes.

Prompt shape tests (direct/reasoning) require no GPU.
Trace prompt tests and extraction tests skip if the tokenizer is unavailable.
"""

from __future__ import annotations

import pytest

from evals.cruxeval.evaluate import (
    extract_answer,
    extract_answer_reasoning,
)
from evals.cruxeval.prompts import (
    make_direct_output_prompt,
    make_reasoning_prompt,
)

# ---------------------------------------------------------------------------
# Sample fixture (CRUXEval 2-shot example)
# ---------------------------------------------------------------------------

CODE = 'def f(s):\n    return s + "a"\n'
INPUT = '"x9j"'
OUTPUT = '"x9ja"'


# ---------------------------------------------------------------------------
# Prompt shape tests (no tokenizer needed)
# ---------------------------------------------------------------------------


def test_direct_prompt():
    prompt = make_direct_output_prompt(CODE, INPUT)
    expected = (
        "You are given a Python function and an assertion containing an input to the function. "
        "Complete the assertion with a literal (no unsimplified expressions, no function calls) "
        "containing the output when executing the provided code on the given input, even if the "
        "function is incorrect or incomplete. Do NOT output any extra information. Provide the full "
        "assertion with the correct output in [ANSWER] and [/ANSWER] tags, following the examples.\n"
        "\n"
        "[PYTHON]\n"
        "def f(n):\n"
        "    return n\n"
        "assert f(17) == ??\n"
        "[/PYTHON]\n"
        "[ANSWER]\n"
        "assert f(17) == 17\n"
        "[/ANSWER]\n"
        "\n"
        "[PYTHON]\n"
        'def f(s):\n'
        '    return s + "a"\n'
        'assert f("x9j") == ??\n'
        "[/PYTHON]\n"
        "[ANSWER]\n"
        'assert f("x9j") == "x9ja"\n'
        "[/ANSWER]\n"
        "\n"
        "[PYTHON]\n"
        'def f(s):\n'
        '    return s + "a"\n'
        "\n"
        'assert f("x9j") == ??\n'
        "[/PYTHON]\n"
        "[ANSWER]\n"
    )
    assert prompt == expected

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def test_reasoning_prompt():
    prompt = make_reasoning_prompt(CODE, INPUT)
    direct = make_direct_output_prompt(CODE, INPUT)
    pre_answer = direct[: -len("[ANSWER]\n")]
    assert prompt == pre_answer + "<think>\n"


@pytest.fixture
def tokenizer(tokenizer_path):
    from cwm.text.tokenizers import build_tokenizer
    return build_tokenizer("cwm_instruct", tokenizer_path)


# ---------------------------------------------------------------------------
# Extraction tests
# ---------------------------------------------------------------------------


def test_extract_direct():
    gen = 'assert f("x9j") == "x9ja"\n[/ANSWER]'
    assert extract_answer(gen, INPUT) == '"x9ja"'


def test_extract_reasoning():
    gen = (
        '<think>\nThe function appends "a". So f("x9j") returns "x9ja".\n</think>\n'
        'assert f("x9j") == "x9ja"\n[/ANSWER]'
    )
    assert extract_answer_reasoning(gen, INPUT) == '"x9ja"'

