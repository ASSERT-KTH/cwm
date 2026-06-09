# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Unit tests for the Table 9 (trace-prediction analysis) reproduction.

These are GPU- and tokenizer-free: they exercise the ground-truth tracer, the
generated-trace parser, and the metric computation. The key invariant is that a
ground-truth trace, rendered to the wire format and parsed back, must score a
perfect 100% on every Table 9 component.
"""

from __future__ import annotations

from dataset.cruxeval.ground_truth import (
    ground_truth_trace,
    make_trace_context,
    render_value,
)
from evals.trace_analysis.metrics import compute_trace_metrics
from dataset.cruxeval.trace_format import (
    TraceEvent,
    parse_generated_trace,
    render_frames_to_generation,
)

# CRUXEval 2-shot example: f(s) = s + "a".
CODE = 'def f(s):\n    return s + "a"\n'
INPUT = '"x9j"'
OUTPUT = '"x9ja"'


def test_make_context_matches_prompt_convention():
    ctx = make_trace_context(CODE, INPUT)
    assert "# << START_OF_TRACE" in ctx
    assert "def main():" in ctx
    assert f"return f({INPUT})" in ctx


def test_render_value():
    # CWM uses Python repr semantics (confirmed against real generations):
    # single-quoted strings, parenthesized tuples, bare ints.
    assert render_value("x9ja") == "'x9ja'"
    assert render_value(17) == "17"
    assert render_value([1, 2]) == "[1, 2]"
    assert render_value(None) == "None"
    assert render_value((4, 1)) == "(4, 1)"
    assert render_value([(4, 1), (2, 3)]) == "[(4, 1), (2, 3)]"


def test_ground_truth_trace_basic():
    # align_to_prompt=False keeps the raw trace including the entry call(main).
    frames, err = ground_truth_trace(CODE, INPUT, align_to_prompt=False)
    assert err is None
    events = [f.event for f in frames]
    assert events[0] == TraceEvent.CALL
    assert TraceEvent.RETURN in events
    sources = [f.source for f in frames]
    assert any(s.startswith("def main()") for s in sources)
    assert any(s.startswith("def f(") for s in sources)
    # The final frame is main returning the output value (repr -> single quotes).
    last_return = [f for f in frames if f.event == TraceEvent.RETURN][-1]
    assert last_return.arg == "'x9ja'"


def test_ground_truth_aligned_drops_entry_call():
    """Default alignment drops the seeded call(main) frame; the first frame is
    the line that calls f, matching what the model generates."""
    frames, _ = ground_truth_trace(CODE, INPUT)
    assert frames[0].event == TraceEvent.LINE
    assert frames[0].source.strip().startswith("return f(")
    assert not any(s.source.startswith("def main()") for s in frames)


def test_roundtrip_is_perfect():
    """A GT trace rendered to wire format must parse to a perfect score."""
    gt_frames, _ = ground_truth_trace(CODE, INPUT)
    generation = render_frames_to_generation(gt_frames)
    pred_frames, well_formed = parse_generated_trace(generation)
    assert well_formed
    m = compute_trace_metrics(gt_frames, pred_frames, well_formed)
    assert m.valid_trace_format == 1.0
    assert m.state_exact_match == 1.0
    assert m.action_exact_match == 1.0
    assert m.valid_json_format == 1.0
    assert m.key_match == 1.0
    assert m.key_value_match == 1.0


def test_diff_based_locals():
    """A variable that changes across lines is shown with its value; an
    unchanged one is rendered as the diff placeholder."""
    code = "def f(x):\n    y = x + 1\n    z = y\n    y = y + 1\n    return y + z\n"
    frames, err = ground_truth_trace(code, "10")
    assert err is None
    # Collect line frames inside f (those that have y in locals).
    line_states = [
        f.locals for f in frames if f.has_locals and f.locals and "y" in f.locals
    ]
    # On the line after `z = y`, y is unchanged -> placeholder "..".
    assert any(s.get("y") == ".." for s in line_states), line_states


def test_imperfect_prediction_scores_partial():
    """A prediction that drops the final frame should not score a perfect
    action match but should still be counted."""
    gt_frames, _ = ground_truth_trace(CODE, INPUT)
    generation = render_frames_to_generation(gt_frames[:-1])  # drop last frame
    pred_frames, well_formed = parse_generated_trace(generation)
    m = compute_trace_metrics(gt_frames, pred_frames, well_formed)
    assert m.action_exact_match is not None
    assert m.action_exact_match < 1.0


def test_malformed_generation_invalid_format():
    gen = "this is not a trace at all"
    pred_frames, well_formed = parse_generated_trace(gen)
    assert not well_formed
    assert pred_frames == []


def test_token_length_stats():
    gt_frames, _ = ground_truth_trace(CODE, INPUT)
    generation = render_frames_to_generation(gt_frames)
    pred_frames, well_formed = parse_generated_trace(generation)
    # Use a trivial whitespace token counter as a stand-in for the tokenizer.
    m = compute_trace_metrics(
        gt_frames, pred_frames, well_formed, token_len=lambda s: max(1, len(s) // 2)
    )
    assert m.avg_state_tokens is not None and m.avg_state_tokens > 0
    assert m.avg_action_tokens is not None and m.avg_action_tokens > 0
