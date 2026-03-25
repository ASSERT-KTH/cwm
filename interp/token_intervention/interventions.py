"""Token-level causal interventions on CWM execution traces.

All four interventions share the same primitive:
  generate baseline trace → find intervention point → inject modified tokens → re-generate

Trace format (from observed data):
  <|line_sep|>{"var": "val", "other": ".."}<|action_sep|>    CODE_LINE\n<|frame_sep|>
  <|return_sep|><|action_sep|>    return expr\n<|arg_sep|>"VALUE"<|frame_sep|>
  <|call_sep|>{"arg": "val"}<|action_sep|>def func():\n<|frame_sep|>

  ".." means the variable is unchanged since the last step shown.
"""

from __future__ import annotations

import ast
import json
import re
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from cwm.text.tokenizers import CWMInstructTokenizer

LINE_SEP   = "<|line_sep|>"
ACTION_SEP = "<|action_sep|>"
FRAME_SEP  = "<|frame_sep|>"
RETURN_SEP = "<|return_sep|>"
ARG_SEP    = "<|arg_sep|>"


def _build_prefix(
    prompt_tokens: list[int],
    generation: str,
    cut_at: int,
    injection: str,
    tokenizer: "CWMInstructTokenizer",
) -> list[int]:
    """Cut generation at cut_at, append injection, re-encode, prepend prompt."""
    new_suffix = generation[:cut_at] + injection
    return prompt_tokens + tokenizer.encode(new_suffix)


# ── Experiment 1: Variable state injection ────────────────────────────────────
#
# Each trace step carries a JSON state dict: {"x": "5", "y": ".."}
# We find the first step where `varname` has a concrete value (not ".."),
# replace it in the JSON, and let the model continue from the modified state.
#
# Question: does the rest of the trace (and the final answer) reflect the new value?

def intervene_variable(
    prompt_tokens: list[int],
    generation: str,
    tokenizer: "CWMInstructTokenizer",
    varname: str,
    new_value: str,
    occurrence: int = 0,
) -> Optional[list[int]]:
    pattern = re.compile(re.escape(LINE_SEP) + r"(\{[^}]+\})" + re.escape(ACTION_SEP))
    hits = []
    for m in pattern.finditer(generation):
        try:
            state = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if varname in state and state[varname] != "..":
            hits.append((m, state))

    if occurrence >= len(hits):
        return None

    m, state = hits[occurrence]
    new_state = {**state, varname: new_value}
    # Cut right after LINE_SEP, inject new JSON + ACTION_SEP so the code line follows
    cut_at = m.start() + len(LINE_SEP)
    injection = json.dumps(new_state, separators=(", ", ": ")) + ACTION_SEP
    # Keep the original code line (everything from ACTION_SEP to end of m)
    injection += generation[m.end(): generation.find(FRAME_SEP, m.end()) + len(FRAME_SEP)]
    return _build_prefix(prompt_tokens, generation, cut_at, injection, tokenizer)


# ── Experiment 2: Return value injection ──────────────────────────────────────
#
# The inner function return has the form:
#   <|return_sep|><|action_sep|>    return EXPR\n<|arg_sep|>"VALUE"<|frame_sep|>
# We keep the prefix up to (and including) <|arg_sep|>, inject a wrong value,
# then close with <|frame_sep|> and let the model continue (it will then generate
# the outer/main() return, which should either trust or re-derive the value).
#
# Question: does the outer return follow the injected inner return?

def intervene_return(
    prompt_tokens: list[int],
    generation: str,
    tokenizer: "CWMInstructTokenizer",
    injected_value: str,
    which: str = "inner",  # "inner" = first return, "outer" = last return
) -> Optional[list[int]]:
    positions = [m.start() for m in re.finditer(re.escape(RETURN_SEP), generation)]
    if not positions:
        return None

    pos = positions[0] if which == "inner" else positions[-1]
    after = generation[pos:]
    arg_idx = after.find(ARG_SEP)
    if arg_idx == -1:
        return None

    # Cut right after ARG_SEP and inject new value + FRAME_SEP
    cut_at = pos + arg_idx + len(ARG_SEP)
    injection = f'"{injected_value}"{FRAME_SEP}'
    return _build_prefix(prompt_tokens, generation, cut_at, injection, tokenizer)


# ── Experiment 3: Branch flip ─────────────────────────────────────────────────
#
# Branches are implicit in the trace: whichever line executes after `if COND:`
# reveals which branch was taken.  We use the code AST to find if/else pairs,
# check which branch the trace actually took, and swap it for the other branch's
# first line.
#
# Question: does the model update subsequent variable states and the final answer
# to reflect the new branch, or does it ignore the injected branch?

def _get_branch_alternatives(code: str) -> list[tuple[str, str, str]]:
    """
    Return list of (if_line, then_first_line, else_first_line) using original
    source indentation, for every if/else (or if/elif) node in the code.
    """
    lines = code.splitlines()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    results = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not node.orelse:
            continue
        if_line   = lines[node.lineno - 1].rstrip()
        then_line = lines[node.body[0].lineno - 1].rstrip()
        else_line = lines[node.orelse[0].lineno - 1].rstrip()
        results.append((if_line, then_line, else_line))
    return results


def intervene_branch(
    prompt_tokens: list[int],
    generation: str,
    tokenizer: "CWMInstructTokenizer",
    code: str,
    occurrence: int = 0,
) -> Optional[list[int]]:
    alternatives = _get_branch_alternatives(code)
    if not alternatives:
        return None

    # For each if/else pair, look for the if_line in the trace and determine
    # which branch was taken; then inject the other branch's first line.
    hits = []
    for if_line, then_line, else_line in alternatives:
        if_marker = ACTION_SEP + if_line + "\n" + FRAME_SEP
        for m in re.finditer(re.escape(if_marker), generation):
            rest = generation[m.end():]
            # Parse the next step: LINE_SEP + JSON + ACTION_SEP + taken_line
            step = re.match(
                re.escape(LINE_SEP) + r"(\{[^}]+\})" + re.escape(ACTION_SEP), rest
            )
            if not step:
                continue
            taken_start = m.end() + step.end()
            taken_end   = generation.find(FRAME_SEP, taken_start)
            if taken_end == -1:
                continue
            taken_line = generation[taken_start:taken_end].rstrip()
            # Determine which branch was taken and what the alternative is
            if taken_line.strip() == then_line.strip():
                alt_line = else_line
            elif taken_line.strip() == else_line.strip():
                alt_line = then_line
            else:
                continue  # unrecognised branch (e.g. loop/multi-line body)
            hits.append((taken_start, alt_line))

    if occurrence >= len(hits):
        return None

    cut_at, alt_line = hits[occurrence]
    injection = alt_line + "\n" + FRAME_SEP
    return _build_prefix(prompt_tokens, generation, cut_at, injection, tokenizer)


# ── Experiment 4: Cross-sample trace corruption ───────────────────────────────
#
# Inject the execution trace from a *different* sample (sample_A) as the prefix
# for a new prompt (sample_B).  The model must generate a continuation given a
# trace that doesn't match the code.
#
# Question: does the model follow the foreign trace, or re-derive from sample_B's code?
# Variant A (truncate=True):  only inject up to the inner ARG_SEP value, so the
#   model must still "close" the trace and produce an outer return.
# Variant B (truncate=False): inject the entire trace including EOS — the model
#   gets a complete but wrong trace and must decide what to output.

def intervene_corrupt_trace(
    prompt_tokens_b: list[int],
    generation_a: str,
    tokenizer: "CWMInstructTokenizer",
    truncate_at_inner_return: bool = True,
) -> list[int]:
    trace = generation_a
    if truncate_at_inner_return:
        # Keep up to and including the first inner ARG_SEP value + FRAME_SEP
        m = re.search(
            re.escape(ARG_SEP) + r'".+?"' + re.escape(FRAME_SEP), trace
        )
        if m:
            trace = trace[: m.end()]
    return prompt_tokens_b + tokenizer.encode(trace)
