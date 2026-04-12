# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Stage classification and label-array construction for SWEbench trajectories.

Each SWEbench trajectory consists of a sequence of (model-action, tool-result)
turns.  This module provides two things:

1. classify_tool_call() — maps a tool call to one of five interpretability stages:
       exploration | testing | editing | submission | other-bash

2. build_label_arrays() — builds probe-training label arrays aligned to the
   captured activation positions (one entry per captured token, not per turn).
"""

from __future__ import annotations

# Commands whose first word unambiguously indicates read-only file inspection.
_EXPLORATION_FIRST_WORDS = {
    "cat", "ls", "find", "grep", "head", "tail", "less", "more",
    "wc", "stat", "file", "pwd", "which", "type", "diff", "echo",
}


def classify_tool_call(tool_name: str, tool_input: str) -> str:
    """Classify a single tool call into an interpretability stage.

    Args:
        tool_name:  One of "bash", "edit", "create", "submit".
        tool_input: Raw text content of the tool call (command string for bash,
                    file path + content for edit/create, empty for submit).

    Returns:
        One of: "exploration", "testing", "editing", "submission", "other-bash".
    """
    if tool_name in ("edit", "create"):
        return "editing"
    if tool_name == "submit":
        return "submission"
    if tool_name == "bash":
        cmd = tool_input.strip()
        # Testing: any invocation of pytest or python -m pytest / python -m test
        if "pytest" in cmd:
            return "testing"
        if "python" in cmd and ("test" in cmd.lower() or "-m test" in cmd):
            return "testing"
        # Exploration: command starts with a known read-only tool
        first_word = cmd.split()[0] if cmd.split() else ""
        if first_word in _EXPLORATION_FIRST_WORDS:
            return "exploration"
        return "other-bash"
    return "other-bash"


def build_label_arrays(
    turn_log: list[dict],
    turn_indices: list[int],
    positions: list[int],
    outcome: bool,
) -> dict[str, list]:
    """Build probe-training label arrays aligned to captured activation positions.

    Args:
        turn_log:     List of dicts, one per turn:
                          {"turn_idx": int, "tool_name": str,
                           "tool_input": str, "stage": str}
        turn_indices: Parallel to `positions` — which turn each capture belongs to.
        positions:    Absolute token positions of each captured activation.
        outcome:      Whether the trajectory's final patch passed the test suite.

    Returns:
        Dict of label arrays, each of length len(positions):
            "pass"      — int (0 or 1), same value everywhere (trajectory-level)
            "stage"     — str, stage of the turn the position belongs to
            "turn_idx"  — int, turn index of each position
    """
    pass_label = int(outcome)
    turn_to_stage: dict[int, str] = {
        entry["turn_idx"]: entry["stage"] for entry in turn_log
    }
    return {
        "pass": [pass_label] * len(positions),
        "stage": [turn_to_stage.get(t, "unknown") for t in turn_indices],
        "turn_idx": list(turn_indices),
    }
