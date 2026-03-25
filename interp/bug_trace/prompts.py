"""Prompt builders for bug-fixing experiments.

Track A: NL reasoning mode — model reasons in <think> and outputs the fix.
Track B: trace_full mode — model traces the (buggy) code execution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cwm.text.tokenizers import CWMInstructTokenizer

# Re-export for convenience
from evals.cruxeval.prompts import (
    REASONING_SYSTEM_PROMPT,
    make_trace_full_prompt_tokens,
)

BUG_FIX_SYSTEM_PROMPT = (
    "You are an expert Python programmer and debugger. "
    "You always reason carefully before responding, using the following format:\n\n"
    "<think>\nyour step-by-step analysis of the bug\n</think>\n"
    "your corrected Python function (only the function, no explanation)"
)


def make_bug_fix_prompt_tokens(
    buggy_code: str,
    input_str: str,
    wrong_output: str,
    correct_output: str,
    tokenizer: "CWMInstructTokenizer",
) -> list[int]:
    """Build a Track-A bug-fixing prompt using <think> reasoning.

    The model is expected to:
    1. Reason inside <think>...</think> about what the bug is
    2. Output the corrected function after </think>

    Format:
      [BOS]
      <|start_header_id|>system<|end_header_id|>\\n\\n{BUG_FIX_SYSTEM_PROMPT}<|eot_id|>
      <|start_header_id|>user<|end_header_id|>\\n\\n{user_msg}<|eot_id|>
      <|start_header_id|>assistant<|end_header_id|>\\n\\n<think>\\n
    """
    user_msg = (
        f"The following Python function has a bug.\n\n"
        f"```python\n{buggy_code}\n```\n\n"
        f"When called as `f({input_str})`, it returns `{wrong_output}` "
        f"but the correct output should be `{correct_output}`.\n\n"
        f"Identify the bug and provide the corrected function."
    )

    tokens = [tokenizer.bos_id]
    # System
    tokens += [tokenizer.header_start_id]
    tokens += tokenizer.encode("system")
    tokens += [tokenizer.header_end_id]
    tokens += tokenizer.encode("\n\n")
    tokens += tokenizer.encode(BUG_FIX_SYSTEM_PROMPT)
    tokens += [tokenizer.eot_id]
    # User
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
    tokens += tokenizer.think_token_ids  # "<think>\n"
    return tokens


def make_bug_trace_prompt_tokens(
    buggy_code: str,
    input_str: str,
    tokenizer: "CWMInstructTokenizer",
) -> list[int]:
    """Track-B: trace_full prompt on the buggy code.

    Identical to make_trace_full_prompt_tokens but named for clarity.
    The model will trace the buggy execution, potentially diverging from
    the correct trace at the mutation point.
    """
    return make_trace_full_prompt_tokens(buggy_code, input_str, tokenizer)
