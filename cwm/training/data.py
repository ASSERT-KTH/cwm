# Copyright (c) Meta Platforms, Inc. and affiliates.

"""CODI training data: trace -> (input_ids, labels), train/val split, then a
length-bucketed DP-sharded batch sampler, padded collate, and DataLoader builder.
``dataset`` only switches sources and generates traces; tokenization, the CODI
teacher-forcing mask, and the splits live here (the consumer), selected by
``data_source``.
"""

from __future__ import annotations

import logging
from functools import partial

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler

from cwm.training.distributed import DistState
from dataset.ground_truth import clean_trace, clean_traces
from dataset.sources import load_rows

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


def _to_example(out) -> tuple[list[int], list[int]] | None:
    """CODI teacher-forcing mask over a ``clean_trace`` result (None passes through):
    prompt tokens are ``IGNORE_INDEX``, trace tokens supervised."""
    if out is None:
        return None
    prompt_ids, trace_ids = out
    return prompt_ids + trace_ids, [IGNORE_INDEX] * len(prompt_ids) + trace_ids


def build_example(
    code: str, input_str: str, tokenizer, *, max_seq_len: int
) -> tuple[list[int], list[int]] | None:
    """``(input_ids, labels)`` for one CODI sample, or None to skip. Filtering and
    tokenization live in ``clean_trace`` (dataset layer); here we only add the mask."""
    return _to_example(clean_trace(code, input_str, tokenizer, max_seq_len=max_seq_len))


def build_dataset(
    rows, tokenizer, *, n_samples: int = -1, max_seq_len: int = 8192, workers: int = 0
) -> list[tuple[list[int], list[int]]]:
    """Tokenized CODI traces from ``rows`` ({code, input}). ``n_samples<=0`` uses all rows.
    ``workers>1`` parallelizes the one-off filter+tokenize build (see ``clean_traces``)."""
    if n_samples > 0:
        rows = rows[:n_samples]
    outs = clean_traces(rows, tokenizer, max_seq_len=max_seq_len, workers=workers)
    return [ex for ex in map(_to_example, outs) if ex is not None]


def collate_codi_batch(batch, pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(ids) for ids, _ in batch)
    input_ids, labels, attn = [], [], []
    for ids, lab in batch:
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(lab + [IGNORE_INDEX] * pad)
        attn.append([1] * len(ids) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attn),
    }


class MegabatchSortishSampler(Sampler[list[int]]):
    """Near-i.i.d. length-bucketed batch sampler.

    Shuffle -> sort within ``batch_size*megabatch_mult`` megabatches (packs
    similar lengths, cuts padding) -> reshuffle batch order (no curriculum).
    DP-sharded with per-window load balancing (see ``_build``): equal batch
    count and near-equal work per grad-accum window (DDP-safe).
    ``max_batch_tokens>0`` sizes batches by padded-token budget instead of a
    fixed count, bounding peak memory on long traces (else OOM).
    """

    def __init__(
        self,
        lengths: list[int],
        *,
        batch_size: int,
        grad_accum_steps: int = 1,
        megabatch_mult: int = 8,
        max_batch_tokens: int = 0,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        self.lengths = lengths
        self.batch_size = batch_size
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.megabatch_mult = max(1, megabatch_mult)
        self.max_batch_tokens = max(0, max_batch_tokens)
        self.num_replicas = max(1, num_replicas)
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        # cache: batch count varies per epoch, keep __len__/__iter__ in sync
        self._batches = self._build(0)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self._batches = self._build(epoch)

    def _split_megabatch(self, mb: list[int]) -> list[list[int]]:
        """Slice a length-descending megabatch into token-budget batches."""
        if not self.max_batch_tokens:
            return [
                mb[j : j + self.batch_size] for j in range(0, len(mb), self.batch_size)
            ]
        out: list[list[int]] = []
        j = 0
        while j < len(mb):
            head = self.lengths[mb[j]]  # longest remaining = padded width
            cap = min(self.batch_size, max(1, self.max_batch_tokens // head))
            out.append(mb[j : j + cap])
            j += cap
        return out

    def _build(self, epoch: int) -> list[list[int]]:
        g = torch.Generator().manual_seed(self.seed + epoch)
        n = len(self.lengths)
        order = (
            torch.randperm(n, generator=g).tolist() if self.shuffle else list(range(n))
        )
        mb_size = self.batch_size * self.megabatch_mult
        batches: list[list[int]] = []
        for i in range(0, n, mb_size):
            mb = sorted(
                order[i : i + mb_size], key=lambda idx: self.lengths[idx], reverse=True
            )
            batches.extend(self._split_megabatch(mb))
        if self.shuffle:
            batches = [
                batches[k] for k in torch.randperm(len(batches), generator=g).tolist()
            ]

        if self.num_replicas > 1 and batches:
            # Per grad-accum window (R*N consecutive batches): greedily split into
            # R bins of N, equalizing token load so ranks finish together (DDP).
            R, N = self.num_replicas, self.grad_accum_steps
            win = R * N
            rem = len(batches) % win
            if rem:  # cycle-pad to whole windows -> equal count per rank
                batches += [batches[k % len(batches)] for k in range(win - rem)]
            load = lambda b: sum(self.lengths[i] for i in b)
            mine: list[list[int]] = []
            for w in range(0, len(batches), win):
                bins: list[list[int]] = [[] for _ in range(R)]
                sums = [0] * R
                for b in sorted(batches[w : w + win], key=load, reverse=True):
                    r = min(
                        (i for i in range(R) if len(bins[i]) < N), key=lambda i: sums[i]
                    )
                    bins[r].append(b)
                    sums[r] += load(b)
                mine.extend(bins[self.rank])
            batches = mine
        return batches

    def __iter__(self):
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)


def build_codi_dataloader(
    *,
    tokenizer,
    pad_id: int,
    dist_state: DistState,
    n_samples: int,
    max_seq_len: int,
    batch_size: int,
    grad_accum_steps: int = 1,
    num_workers: int,
    seed: int,
    megabatch_mult: int = 8,
    max_batch_tokens: int = 0,
    build_workers: int = 0,
    data_source: list[str],
) -> tuple[DataLoader, MegabatchSortishSampler]:
    if dist_state.is_rank_zero or not dist_state.enabled:
        dataset = build_dataset(
            load_rows(data_source), tokenizer, n_samples=n_samples,
            max_seq_len=max_seq_len, workers=build_workers,
        )
    else:
        dataset = None

    if dist_state.enabled:
        obj = [dataset]
        dist.broadcast_object_list(obj, src=0, device=dist_state.device)
        dataset = obj[0]

    lengths = [len(ids) for ids, _ in dataset]
    # max_batch_tokens: 0 = fixed batch_size; >0 = padded-token budget per batch
    batch_sampler = MegabatchSortishSampler(
        lengths,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        megabatch_mult=megabatch_mult,
        max_batch_tokens=max_batch_tokens,
        num_replicas=dist_state.dp_size if dist_state.dp_enabled else 1,
        rank=dist_state.dp_rank if dist_state.dp_enabled else 0,
        shuffle=True,
        seed=seed,
    )
    if dist_state.is_rank_zero or not dist_state.enabled:
        batches = batch_sampler._batches
        sizes = [len(b) for b in batches]
        loads = [max(lengths[i] for i in b) * len(b) for b in batches]
        logger.info(
            "codi batch budget: max_batch_tokens=%d  batches/rank=%d  rows/batch=%d-%d  "
            "peak padded load=%d tok",
            max_batch_tokens,
            len(sizes),
            min(sizes),
            max(sizes),
            max(loads),
        )

    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=partial(collate_codi_batch, pad_id=pad_id),
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader, batch_sampler
