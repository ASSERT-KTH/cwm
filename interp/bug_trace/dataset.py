"""Load and partition the (original, buggy) pair dataset.

The dataset is a list of dicts, each with:
  pair_id, original_id, original_code, input_str, correct_output,
  buggy_code, wrong_output, mutation_type

We expand it into a flat list of "samples", each tagged with is_buggy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BugSample:
    sample_id: str        # unique ID for this extraction sample
    pair_id: str          # links original and all its buggy variants
    original_id: str      # CRUXEval sample ID
    is_buggy: bool
    mutation_type: str    # "none" for original
    code: str             # the program text (original or buggy)
    input_str: str
    correct_output: str   # always the correct expected output
    wrong_output: str     # == correct_output for originals (unused)


def load_pairs(pairs_path: str | Path) -> list[dict]:
    with Path(pairs_path).open() as f:
        return json.load(f)


def pairs_to_samples(pairs: list[dict], include_originals: bool = True) -> list[BugSample]:
    """Expand each pair into (original, buggy) BugSamples.

    Returns a flat list. Originals appear once per unique original_id if
    include_originals=True.
    """
    seen_originals: set[str] = set()
    samples: list[BugSample] = []

    for p in pairs:
        oid = p["original_id"]
        # Original (deduplicated)
        if include_originals and oid not in seen_originals:
            seen_originals.add(oid)
            samples.append(
                BugSample(
                    sample_id=f"orig__{oid}",
                    pair_id=p["pair_id"],
                    original_id=oid,
                    is_buggy=False,
                    mutation_type="none",
                    code=p["original_code"],
                    input_str=p["input_str"],
                    correct_output=p["correct_output"],
                    wrong_output=p["correct_output"],
                )
            )
        # Buggy variant
        samples.append(
            BugSample(
                sample_id=f"bug__{p['pair_id']}",
                pair_id=p["pair_id"],
                original_id=oid,
                is_buggy=True,
                mutation_type=p["mutation_type"],
                code=p["buggy_code"],
                input_str=p["input_str"],
                correct_output=p["correct_output"],
                wrong_output=p["wrong_output"],
            )
        )

    return samples
