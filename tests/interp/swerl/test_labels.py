"""Tests for interp/swerl/labels.py.

Spec: classify_tool_call maps (tool_name, tool_input) → stage string.
      build_label_arrays produces aligned label arrays for probe training.
"""

import pytest

from interp.swerl.labels import build_label_arrays, classify_tool_call


# ---------------------------------------------------------------------------
# classify_tool_call — stage taxonomy
# ---------------------------------------------------------------------------


def test_bash_cat_is_exploration():
    assert classify_tool_call("bash", "cat /testbed/django/utils/decorators.py") == "exploration"


def test_bash_ls_is_exploration():
    assert classify_tool_call("bash", "ls -la /testbed/") == "exploration"


def test_bash_find_is_exploration():
    assert classify_tool_call("bash", "find . -name '*.py' -path '*/tests/*'") == "exploration"


def test_bash_grep_is_exploration():
    assert classify_tool_call("bash", "grep -r 'def test_' /testbed/tests/") == "exploration"


def test_bash_head_is_exploration():
    assert classify_tool_call("bash", "head -40 /testbed/README.rst") == "exploration"


def test_bash_pytest_is_testing():
    assert classify_tool_call("bash", "cd /testbed && python -m pytest tests/basic/") == "testing"


def test_bash_pytest_direct_is_testing():
    assert classify_tool_call("bash", "pytest django/tests/utils_tests/ -x -v") == "testing"


def test_bash_python_m_test_is_testing():
    assert classify_tool_call("bash", "python -m test regrtest -v") == "testing"


def test_edit_is_editing():
    assert classify_tool_call("edit", "path=/testbed/django/utils/decorators.py\n...") == "editing"


def test_create_is_editing():
    assert classify_tool_call("create", "path=/testbed/fix.py\ncontent=...") == "editing"


def test_submit_is_submission():
    assert classify_tool_call("submit", "") == "submission"


def test_bash_pip_install_is_other():
    assert classify_tool_call("bash", "pip install -e .") == "other-bash"


def test_bash_git_log_is_other():
    assert classify_tool_call("bash", "git log --oneline -5") == "other-bash"


def test_bash_python_script_is_other():
    # Running a non-test script — not read-only, not pytest
    assert classify_tool_call("bash", "python reproduce_bug.py") == "other-bash"


# ---------------------------------------------------------------------------
# build_label_arrays — label alignment
# ---------------------------------------------------------------------------


def _make_turn_log(*stages):
    """Helper: build a turn_log with the given stages."""
    tools = {"exploration": "bash", "testing": "bash", "editing": "edit", "other-bash": "bash"}
    return [
        {"turn_idx": i, "tool_name": tools.get(s, "bash"), "tool_input": "cmd", "stage": s}
        for i, s in enumerate(stages)
    ]


def test_build_label_arrays_length_matches_positions():
    """Label arrays are parallel to captured positions."""
    turn_log = _make_turn_log("exploration", "testing", "editing")
    turn_indices = [0, 0, 1, 2]  # 2 captures in turn 0, 1 in turn 1, 1 in turn 2
    positions = [10, 20, 50, 80]

    labels = build_label_arrays(turn_log, turn_indices, positions, outcome=True)

    assert len(labels["pass"]) == 4
    assert len(labels["stage"]) == 4
    assert len(labels["turn_idx"]) == 4


def test_build_label_arrays_pass_is_trajectory_level():
    """'pass' is the same value at every position — it's the final outcome, not per-step."""
    turn_log = _make_turn_log("exploration")
    labels_pass = build_label_arrays(turn_log, [0, 0, 0], [0, 1, 2], outcome=True)
    labels_fail = build_label_arrays(turn_log, [0, 0, 0], [0, 1, 2], outcome=False)

    assert labels_pass["pass"] == [1, 1, 1]
    assert labels_fail["pass"] == [0, 0, 0]


def test_build_label_arrays_stage_aligns_with_turn():
    """stage[i] reflects the stage of the turn that position i belongs to."""
    turn_log = _make_turn_log("exploration", "testing", "editing")
    turn_indices = [0, 0, 1, 2]

    labels = build_label_arrays(turn_log, turn_indices, [0, 1, 2, 3], outcome=True)

    assert labels["stage"] == ["exploration", "exploration", "testing", "editing"]


def test_build_label_arrays_turn_idx_preserved():
    """turn_idx array is a copy of the input turn_indices."""
    turn_log = _make_turn_log("exploration", "testing")
    turn_indices = [0, 0, 1, 1, 1]

    labels = build_label_arrays(turn_log, turn_indices, list(range(5)), outcome=False)

    assert labels["turn_idx"] == [0, 0, 1, 1, 1]
