# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Canonical CRUXEval-O prompt templates.
Ported verbatim from https://github.com/facebookresearch/cruxeval/blob/main/prompts.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cwm.text.tokenizers import CWMInstructTokenizer

REASONING_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. You always reason before responding, "
    "using the following format:\n\n"
    "<think>\n"
    "your internal reasoning\n"
    "</think>\n"
    "your external response"
)


def make_direct_output_prompt(code: str, input: str) -> str:
    return f"""You are given a Python function and an assertion containing an input to the function. Complete the assertion with a literal (no unsimplified expressions, no function calls) containing the output when executing the provided code on the given input, even if the function is incorrect or incomplete. Do NOT output any extra information. Provide the full assertion with the correct output in [ANSWER] and [/ANSWER] tags, following the examples.

[PYTHON]
def f(n):
    return n
assert f(17) == ??
[/PYTHON]
[ANSWER]
assert f(17) == 17
[/ANSWER]

[PYTHON]
def f(s):
    return s + "a"
assert f("x9j") == ??
[/PYTHON]
[ANSWER]
assert f("x9j") == "x9ja"
[/ANSWER]

[PYTHON]
{code}
assert f({input}) == ??
[/PYTHON]
[ANSWER]
"""


def make_reasoning_prompt_tokens(
    code: str, input_str: str, tokenizer: "CWMInstructTokenizer"
) -> list[int]:
    """
    Build a chat-formatted reasoning prompt following the CWM prompting guide.

    Format:
      [BOS]
      <|start_header_id|>system<|end_header_id|>\\n\\n{REASONING_SYSTEM_PROMPT}<|eot_id|>
      <|start_header_id|>user<|end_header_id|>\\n\\n{cruxeval question}<|eot_id|>
      <|start_header_id|>assistant<|end_header_id|>\\n\\n<think>\\n

    The model then generates: reasoning...</think>\\n[ANSWER]\\nassert f(...) == value\\n[/ANSWER]
    """
    base = make_direct_output_prompt(code, input_str)
    assert base.endswith("[ANSWER]\n"), "unexpected prompt suffix"
    # The user message is everything up to (but not including) [ANSWER]\n
    user_msg = base[: -len("[ANSWER]\n")]

    tokens = [tokenizer.bos_id]
    # System message
    tokens += [tokenizer.header_start_id]
    tokens += tokenizer.encode("system")
    tokens += [tokenizer.header_end_id]
    tokens += tokenizer.encode("\n\n")
    tokens += tokenizer.encode(REASONING_SYSTEM_PROMPT)
    tokens += [tokenizer.eot_id]
    # User message
    tokens += [tokenizer.header_start_id]
    tokens += tokenizer.encode("user")
    tokens += [tokenizer.header_end_id]
    tokens += tokenizer.encode("\n\n")
    tokens += tokenizer.encode(user_msg)
    tokens += [tokenizer.eot_id]
    # Assistant header + <think>\n
    tokens += [tokenizer.header_start_id]
    tokens += tokenizer.encode("assistant")
    tokens += [tokenizer.header_end_id]
    tokens += tokenizer.encode("\n\n")
    tokens += tokenizer.think_token_ids
    return tokens

