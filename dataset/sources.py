# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Common entry for training/eval: dataset name(s) -> merged rows.

Add a converted dataset by dropping a folder with a save_to_disk ``./data`` and
listing it in ``_LOCAL``; cruxeval_o has its own loader (Hub fallback).
"""

from __future__ import annotations

import os
from pathlib import Path

_LOCAL = {"mbpp": "MBPP", "humaneval": "HumanEval", "pyx": "PyX"}  # name -> folder, data in ./data


def _load_one(name: str) -> list[dict]:
    key = name.strip().lower()
    if key == "cruxeval_o":
        from dataset.cruxeval_o.dataset import load_rows as _crux

        return _crux()
    if key in _LOCAL:
        from datasets import load_from_disk

        d = os.environ.get(key.upper() + "_DIR") or str(Path(__file__).parent / _LOCAL[key] / "data")
        return list(load_from_disk(d))
    raise ValueError(f"unknown data source {name!r}; pick from {['cruxeval_o', *_LOCAL]}")


def load_rows(sources: list[str]) -> list[dict]:
    """Merged rows for the given dataset names, e.g. ``["mbpp", "humaneval"]``."""
    rows: list[dict] = []
    for name in sources:
        rows += _load_one(name)
    return rows
