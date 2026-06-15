# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
CRUXEval-O row loader.

CRUXEval-O is one peer dataset among {cruxeval_o, MBPP, HumanEval}, reached via
the ``dataset.sources`` registry. This module only knows where CRUXEval-O rows
come from: rows expose the canonical 4 fields ({code, input, output, id}); CODI
tokenization happens later in ``cwm.training.data``.
"""

from __future__ import annotations

import os


def load_rows() -> list[dict]:
    """CRUXEval-O rows ({code, input, output, id}).

    Prefer a local ``save_to_disk`` copy via ``CRUXEVAL_DIR`` (the HF builder's
    FileLock dies on NFS caches); otherwise pull ``cruxeval-org/cruxeval`` from
    the Hub.
    """
    local_dir = os.environ.get("CRUXEVAL_DIR")
    if local_dir and os.path.isdir(local_dir):
        from datasets import load_from_disk

        return list(load_from_disk(local_dir))
    from datasets import load_dataset

    return list(load_dataset("cruxeval-org/cruxeval", split="test"))
