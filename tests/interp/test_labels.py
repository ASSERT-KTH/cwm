"""Tests for trace parsing → abstract property labels."""

import pytest

from interp.probes.labels import (
    event_type_from_token_id,
    extract_labels,
    extract_labels_from_answer,
)


def test_extract_return_type_int():
    labels = extract_labels_from_answer("42")
    assert labels["return_type"] == "int"


def test_extract_return_type_str():
    labels = extract_labels_from_answer('"hello"')
    assert labels["return_type"] == "str"


def test_extract_return_type_list():
    labels = extract_labels_from_answer("[1, 2, 3]")
    assert labels["return_type"] == "list"


def test_extract_return_type_tuple():
    labels = extract_labels_from_answer("(1, 2)")
    assert labels["return_type"] == "tuple"


def test_extract_return_type_bool():
    labels = extract_labels_from_answer("True")
    assert labels["return_type"] == "bool"


def test_extract_return_type_none():
    labels = extract_labels_from_answer("None")
    assert labels["return_type"] == "None"


def test_extract_return_sign_positive():
    labels = extract_labels_from_answer("42")
    assert labels["return_sign"] == "positive"


def test_extract_return_sign_negative():
    labels = extract_labels_from_answer("-7")
    assert labels["return_sign"] == "negative"


def test_extract_return_sign_zero():
    labels = extract_labels_from_answer("0")
    assert labels["return_sign"] == "zero"


def test_extract_return_sign_na_for_string():
    labels = extract_labels_from_answer('"hello"')
    assert labels["return_sign"] == "N/A"


def test_extract_return_sign_na_for_none():
    labels = extract_labels_from_answer("None")
    assert labels["return_sign"] == "N/A"


def test_extract_return_truthy_nonzero():
    assert extract_labels_from_answer("42")["return_truthy"] is True


def test_extract_return_truthy_zero():
    assert extract_labels_from_answer("0")["return_truthy"] is False


def test_extract_return_truthy_empty_string():
    assert extract_labels_from_answer('""')["return_truthy"] is False


def test_extract_return_truthy_empty_list():
    assert extract_labels_from_answer("[]")["return_truthy"] is False


def test_extract_return_truthy_none():
    assert extract_labels_from_answer("None")["return_truthy"] is False


def test_extract_return_length_bin_zero():
    assert extract_labels_from_answer("[]")["return_length_bin"] == "0"


def test_extract_return_length_bin_one():
    assert extract_labels_from_answer("[1]")["return_length_bin"] == "1"


def test_extract_return_length_bin_small():
    assert extract_labels_from_answer("[1,2,3]")["return_length_bin"] == "2-5"


def test_extract_return_length_bin_medium():
    assert extract_labels_from_answer('"hello world!"')["return_length_bin"] == "6-20"


def test_extract_return_length_bin_large():
    assert extract_labels_from_answer('"' + "x" * 25 + '"')["return_length_bin"] == "20+"


def test_extract_return_length_na_for_int():
    assert extract_labels_from_answer("42")["return_length_bin"] == "N/A"


def test_extract_event_type_return():
    assert event_type_from_token_id(102) == "return"


def test_extract_event_type_call():
    assert event_type_from_token_id(103) == "call"


def test_extract_event_type_line():
    assert event_type_from_token_id(104) == "line"


def test_extract_event_type_exception():
    assert event_type_from_token_id(105) == "exception"


def test_extract_labels_full():
    labels = extract_labels(
        generated_text="...",
        token_ids=[100, 101, 102, 106, 100],
        captured_positions=[0, 1, 2, 3, 4],
        correct=True,
        extracted_answer="42",
    )
    assert len(labels["will_be_correct"]) == 5
    assert all(v == 1 for v in labels["will_be_correct"])
    assert labels["return_type"] == ["int"] * 5
    assert labels["return_sign"] == ["positive"] * 5
    # Event types from token IDs at captured positions
    assert labels["trace_event_type"][2] == "return"  # token_id=102


def test_extract_labels_incorrect_sample():
    labels = extract_labels(
        generated_text="",
        token_ids=[100],
        captured_positions=[0],
        correct=False,
        extracted_answer="None",
    )
    assert labels["will_be_correct"] == [0]
