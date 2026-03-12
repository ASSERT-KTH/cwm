"""Dataset over extracted activations for probe training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from interp.probes.labels import extract_labels

# Property → number of classes
PROPERTY_N_CLASSES: dict[str, int] = {
    "will_be_correct": 2,
    "return_type": 7,      # int, str, list, tuple, bool, None, other
    "return_sign": 4,      # positive, negative, zero, N/A
    "return_truthy": 2,
    "return_length_bin": 6,  # 0, 1, 2-5, 6-20, 20+, N/A
    "trace_event_type": 4,  # call, return, line, exception
}

# Ordered class lists for integer encoding
_CLASSES: dict[str, list] = {
    "will_be_correct": [0, 1],
    "return_type": ["int", "str", "list", "tuple", "bool", "None", "other"],
    "return_sign": ["positive", "negative", "zero", "N/A"],
    "return_truthy": [False, True],
    "return_length_bin": ["0", "1", "2-5", "6-20", "20+", "N/A"],
    "trace_event_type": ["call", "return", "line", "exception"],
}


def _label_to_int(prop: str, label: Any) -> int:
    classes = _CLASSES[prop]
    if label in classes:
        return classes.index(label)
    # Fallback: last class index (catch-all)
    return len(classes) - 1


class ProbeDataset(Dataset):
    """Loads (hidden_state, label) pairs from a Phase-1 extraction directory.

    Each item is (tensor[dim], int_label).
    """

    def __init__(
        self,
        extract_dir: str,
        layer: int,
        target_property: str,
        position_filter: str = "all",  # all | return_only | frame_sep_only
    ) -> None:
        self.extract_dir = Path(extract_dir)
        self.layer = layer
        self.target_property = target_property
        self.position_filter = position_filter

        # Read index
        index_path = self.extract_dir / "index.jsonl"
        self._index: list[dict] = []
        with index_path.open() as f:
            for line in f:
                self._index.append(json.loads(line))

        # Build (sample_id, position_idx) pairs
        self._items: list[tuple[str, int]] = []
        self._labels: list[int] = []
        self._activations: list[torch.Tensor] = []

        self._build()

    def _build(self) -> None:
        act_dir = self.extract_dir / "activations"
        for meta in self._index:
            sid = meta["sample_id"]
            pt_path = act_dir / f"{sid}.pt"
            if not pt_path.exists():
                continue

            sample = torch.load(pt_path, map_location="cpu", weights_only=False)
            activations = sample.get("activations", {})
            if self.layer not in activations:
                continue

            h = activations[self.layer]  # [n_pos, dim]
            token_ids = sample.get("token_ids", [])
            captured_positions = sample.get("captured_positions", list(range(h.shape[0])))
            correct = sample.get("correct", False)
            extracted_answer = sample.get("extracted_answer", None)
            generated_text = sample.get("generated_text", "")

            per_pos_labels = extract_labels(
                generated_text=generated_text,
                token_ids=token_ids,
                captured_positions=captured_positions,
                correct=correct,
                extracted_answer=extracted_answer,
            )

            prop_labels = per_pos_labels.get(self.target_property, [])

            for pos_idx in range(h.shape[0]):
                if pos_idx >= len(prop_labels):
                    continue

                # Apply position filter
                if self.position_filter != "all":
                    if pos_idx >= len(captured_positions):
                        continue
                    seq_pos = captured_positions[pos_idx]
                    if pos_idx < len(token_ids):
                        tok = token_ids[seq_pos] if seq_pos < len(token_ids) else -1
                    else:
                        tok = -1
                    if self.position_filter == "return_only" and tok != 102:
                        continue
                    if self.position_filter == "frame_sep_only" and tok != 100:
                        continue

                raw_label = prop_labels[pos_idx]
                label = _label_to_int(self.target_property, raw_label)

                self._activations.append(h[pos_idx])
                self._labels.append(label)

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self._activations[idx], self._labels[idx]

    @property
    def n_classes(self) -> int:
        return PROPERTY_N_CLASSES.get(self.target_property, 2)

    @property
    def dim(self) -> int:
        if self._activations:
            return self._activations[0].shape[-1]
        return 0
