"""Parse CruxEval answers/traces into abstract property labels.

All label extraction is purely textual — no model is needed.
"""

from __future__ import annotations

import ast

# Trace separator token IDs (from CWM spec)
FRAME_SEP_ID = 100
ACTION_SEP_ID = 101
RETURN_SEP_ID = 102
CALL_SEP_ID = 103
LINE_SEP_ID = 104
EXCEPTION_SEP_ID = 105
ARG_SEP_ID = 106
TRACE_CONTEXT_START_ID = 107

_EVENT_TYPE_MAP = {
    RETURN_SEP_ID: "return",
    CALL_SEP_ID: "call",
    LINE_SEP_ID: "line",
    EXCEPTION_SEP_ID: "exception",
    FRAME_SEP_ID: "frame",
    ACTION_SEP_ID: "action",
    ARG_SEP_ID: "arg",
    TRACE_CONTEXT_START_ID: "context",
}


def event_type_from_token_id(token_id: int) -> str:
    """Map a trace separator token ID to its event type string."""
    return _EVENT_TYPE_MAP.get(token_id, "unknown")


def _try_eval(answer: str):
    """Try to evaluate the answer string as a Python literal. Returns (value, ok)."""
    try:
        return ast.literal_eval(answer.strip()), True
    except Exception:
        return None, False


def extract_labels_from_answer(answer: str) -> dict:
    """Extract property labels from a predicted answer string.

    Returns a dict with keys:
      return_type, return_sign, return_truthy, return_length_bin
    """
    val, ok = _try_eval(answer)

    # return_type
    if not ok:
        rtype = "other"
    elif val is None:
        rtype = "None"
    elif isinstance(val, bool):
        rtype = "bool"
    elif isinstance(val, int):
        rtype = "int"
    elif isinstance(val, float):
        rtype = "float"
    elif isinstance(val, str):
        rtype = "str"
    elif isinstance(val, list):
        rtype = "list"
    elif isinstance(val, tuple):
        rtype = "tuple"
    else:
        rtype = "other"

    # return_sign (only for numeric)
    if rtype in ("int", "float") and ok:
        if val > 0:
            rsign = "positive"
        elif val < 0:
            rsign = "negative"
        else:
            rsign = "zero"
    else:
        rsign = "N/A"

    # return_truthy
    if ok:
        rtruthy = bool(val)
    else:
        rtruthy = False

    # return_length_bin (for str, list, tuple)
    if rtype in ("str", "list", "tuple") and ok:
        n = len(val)
        if n == 0:
            rlen_bin = "0"
        elif n == 1:
            rlen_bin = "1"
        elif n <= 5:
            rlen_bin = "2-5"
        elif n <= 20:
            rlen_bin = "6-20"
        else:
            rlen_bin = "20+"
    else:
        rlen_bin = "N/A"

    return {
        "return_type": rtype,
        "return_sign": rsign,
        "return_truthy": rtruthy,
        "return_length_bin": rlen_bin,
    }


def extract_labels(
    generated_text: str,
    token_ids: list[int],
    captured_positions: list[int],
    correct: bool,
    extracted_answer: str | None = None,
) -> dict[str, list]:
    """Extract per-position labels for all captured positions.

    Returns {property_name: [label_per_captured_position]}.
    """
    n = len(captured_positions)

    # will_be_correct: same for all positions in this sample
    will_be_correct = [int(correct)] * n

    # trace_event_type: derived from the token ID at each captured position
    trace_event_types = []
    for pos in captured_positions:
        if pos < len(token_ids):
            trace_event_types.append(event_type_from_token_id(token_ids[pos]))
        else:
            trace_event_types.append("unknown")

    # Answer-level labels: same for all positions
    answer_labels: dict = {}
    if extracted_answer is not None:
        answer_labels = extract_labels_from_answer(extracted_answer)
    else:
        answer_labels = {
            "return_type": "other",
            "return_sign": "N/A",
            "return_truthy": False,
            "return_length_bin": "N/A",
        }

    return {
        "will_be_correct": will_be_correct,
        "trace_event_type": trace_event_types,
        "return_type": [answer_labels["return_type"]] * n,
        "return_sign": [answer_labels["return_sign"]] * n,
        "return_truthy": [answer_labels["return_truthy"]] * n,
        "return_length_bin": [answer_labels["return_length_bin"]] * n,
    }
