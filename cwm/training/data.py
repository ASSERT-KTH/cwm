# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Ground-truth execution traces -> (input_ids, labels) for teacher-forcing CODI.

Thin HuggingFace-tokenizer wrapper over the verbatim Table 9 trace generator
(``evals.trace_analysis``). All trace logic is reused: we only build the seeded
prompt, tokenize ``prompt + render_frames_to_generation(frames)``, and mask the
prompt out of the labels (the student is teacher-forced, so labels == input_ids
with the prompt prefix set to ``-100``).
"""

from __future__ import annotations

import logging
from functools import partial

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler

from evals.trace_analysis.ground_truth import ground_truth_trace, make_trace_context
from evals.trace_analysis.trace_format import render_frames_to_generation

from cwm.training.distributed import DistState

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


def _prompt_str(code: str, input_str: str) -> str:
    ctx = make_trace_context(code, input_str)
    return f"<|trace_context_start|>{ctx}<|frame_sep|><|call_sep|>{{}}<|action_sep|>def main():\n<|frame_sep|>"


def build_example(
    code: str, input_str: str, tokenizer, *, max_seq_len: int
) -> tuple[list[int], list[int]] | None:
    """Return ``(input_ids, labels)``, or None to skip (empty / too long).

    A raised program is kept: its EXCEPTION frame is part of the trace to predict.
    ``render_frames_to_generation`` already terminates the trace with ``<|end_of_text|>``.
    """
    frames, _error = ground_truth_trace(code, input_str, align_to_prompt=True)
    if not frames:
        return None
    prompt_ids = [tokenizer.bos_token_id] + tokenizer.encode(
        _prompt_str(code, input_str), add_special_tokens=False
    )
    trace_ids = tokenizer.encode(render_frames_to_generation(frames), add_special_tokens=False)
    input_ids = prompt_ids + trace_ids
    if len(input_ids) > max_seq_len:
        return None
    return input_ids, [IGNORE_INDEX] * len(prompt_ids) + trace_ids


def build_dataset(tokenizer, *, n_samples: int = -1, max_seq_len: int = 8192) -> list[tuple[list[int], list[int]]]:
    """Tokenized CRUXEval-O traces. ``n_samples<=0`` uses all 800."""
    import os

    # Prefer local save_to_disk copy; HF builder FileLock dies on NFS caches.
    local_dir = os.environ.get("CRUXEVAL_DIR")
    if local_dir and os.path.isdir(local_dir):
        from datasets import load_from_disk

        rows = list(load_from_disk(local_dir))
    else:
        from datasets import load_dataset

        rows = list(load_dataset("cruxeval-org/cruxeval", split="test"))
    if n_samples > 0:
        rows = rows[:n_samples]
    examples = (build_example(r["code"], r["input"], tokenizer, max_seq_len=max_seq_len) for r in rows)
    return [ex for ex in examples if ex is not None]


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
    """Length-bucketed batch sampler that keeps most of the random-batching i.i.d.

    Shuffles globally, sorts within megabatches of ``batch_size * megabatch_mult``
    to pack similar-length traces together (cuts padding -> frees memory), then
    shuffles batch order so no length curriculum survives. Larger ``megabatch_mult``
    saves more padding but makes batches more difficulty-homogeneous; small values
    stay close to pure random batching. DP-sharded with per-window load balancing
    (see ``_build``) so every rank yields the same number of batches and finishes
    each grad-accum window with near-equal work (DDP-safe, no straggler).

    When ``max_batch_tokens`` is set, batches are sized by a padded-token budget
    instead of a fixed count: each batch holds at most ``max_batch_tokens //
    longest_row`` rows (still capped at ``batch_size``). Long traces -> smaller
    batches, so peak memory is bounded by the budget rather than by the worst-case
    all-long batch that fixed-size length-sorting produces (which OOMs).
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
        drop_last: bool = False,
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
        self.drop_last = drop_last
        self.epoch = 0
        # Batch count varies per epoch; cache so __len__ and __iter__ stay in sync.
        self._batches = self._build(0)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self._batches = self._build(epoch)

    def _split_megabatch(self, mb: list[int]) -> list[list[int]]:
        """Slice a length-descending megabatch into batches under the token budget."""
        if not self.max_batch_tokens:
            return [mb[j : j + self.batch_size] for j in range(0, len(mb), self.batch_size)]
        out: list[list[int]] = []
        j = 0
        while j < len(mb):
            head = self.lengths[mb[j]]  # longest remaining -> this batch's padded width
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
            mb = sorted(order[i : i + mb_size], key=lambda idx: self.lengths[idx], reverse=True)
            batches.extend(self._split_megabatch(mb))
        if self.drop_last and not self.max_batch_tokens:
            batches = [b for b in batches if len(b) == self.batch_size]
        if self.shuffle:
            batches = [batches[k] for k in torch.randperm(len(batches), generator=g).tolist()]

        if self.num_replicas > 1 and batches:
            # Load-balance per grad-accum window: each window (num_replicas *
            # grad_accum_steps consecutive batches, a random difficulty mix) is
            # greedily split into num_replicas bins of grad_accum_steps each,
            # equalizing each rank's real-token load. All ranks then finish the
            # window together and sync once at its grad all-reduce -> no per-step
            # straggler, and windows stay random (no curriculum at the step level).
            R, N = self.num_replicas, self.grad_accum_steps
            win = R * N
            rem = len(batches) % win
            if rem:
                if self.drop_last:
                    batches = batches[: len(batches) - rem]
                else:  # cycle-pad to whole windows so every rank gets equal count
                    batches += [batches[k % len(batches)] for k in range(win - rem)]
            load = lambda b: sum(self.lengths[i] for i in b)
            mine: list[list[int]] = []
            for w in range(0, len(batches), win):
                bins: list[list[int]] = [[] for _ in range(R)]
                sums = [0] * R
                for b in sorted(batches[w : w + win], key=load, reverse=True):
                    r = min((i for i in range(R) if len(bins[i]) < N), key=lambda i: sums[i])
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
) -> tuple[DataLoader, MegabatchSortishSampler]:
    if dist_state.is_rank_zero or not dist_state.enabled:
        dataset = build_dataset(tokenizer, n_samples=n_samples, max_seq_len=max_seq_len)
    else:
        dataset = None

    if dist_state.enabled:
        obj = [dataset]
        dist.broadcast_object_list(obj, src=0, device=dist_state.device)
        dataset = obj[0]

    lengths = [len(ids) for ids, _ in dataset]
    # max_batch_tokens: 0 = no limit (fixed batch_size); >0 = padded-token budget per batch.
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
            max_batch_tokens, len(sizes), min(sizes), max(sizes), max(loads),
        )

    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=partial(collate_codi_batch, pad_id=pad_id),
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader, batch_sampler
