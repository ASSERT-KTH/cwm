# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.cache_utils import DynamicCache

from cwm.training.codi_config import KdVecs, select_kd_layers

HiddenRequest = Literal["none", "all_layers", "last_layer"]


@dataclass(frozen=True)
class _TextSpan:
    start: int
    end: int
    insert_latent_after: bool

    @property
    def length(self) -> int:
        return self.end - self.start


def _student_spans(
    start_token_id: int,
    end_token_id: int,
    row: list[int],
    valid_len: int,
) -> list[_TextSpan]:
    spans: list[_TextSpan] = []
    prev = 0
    while prev < valid_len:
        start = next(
            (i for i in range(prev, valid_len) if row[i] == start_token_id),
            None,
        )
        if start is None:
            spans.append(_TextSpan(prev, valid_len, False))
            break

        end = next(
            (i for i in range(start + 1, valid_len) if row[i] == end_token_id),
            None,
        )
        if end is None:
            spans.append(_TextSpan(prev, valid_len, False))
            break

        spans.append(_TextSpan(prev, start + 1, True))
        prev = end

    return spans


class _StreamingStudent:

    def __init__(
        self,
        model,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        batch_idx: torch.Tensor,
        teacher_pos: torch.Tensor,
    ) -> None:
        self.model = model
        self.cfg = model.config
        self.input_ids = input_ids
        self.labels = labels
        self.device = input_ids.device
        self.embed = model.input_embeddings
        self.hidden = self.embed.weight.shape[-1]
        self.batch_size = input_ids.shape[0]
        self._pos_dtype = batch_idx.dtype

        self.latent_start_embed = (
            self._token_embed(self.cfg.latent_start_token_id)
            if self.cfg.latent_start_token_id is not None
            else None
        )
        self.latent_end_embed = (
            self._token_embed(self.cfg.latent_end_token_id)
            if self.cfg.latent_end_token_id is not None
            else None
        )
        self.ignore_lab = labels.new_full((1,), self.cfg.ignore_index)

        rows = input_ids.tolist()
        valid_lens = attention_mask.sum(dim=1).tolist()
        self.spans_by_row = [
            _student_spans(
                self.cfg.latent_span_start_token_id,
                self.cfg.latent_span_end_token_id,
                rows[b],
                valid_lens[b],
            )
            for b in range(self.batch_size)
        ]
        self.targets_by_row = [
            teacher_pos[batch_idx == b].tolist() for b in range(self.batch_size)
        ]

        self.proj_device = self.proj_dtype = None
        if model.thought_projector is not None:
            param = next(model.thought_projector.parameters())
            self.proj_device, self.proj_dtype = param.device, param.dtype

        self._kv_chunks: list[tuple[torch.Tensor, ...]] = []
        self.cache_mask = torch.zeros(
            self.batch_size, 0, device=self.device, dtype=torch.long
        )
        self.prev_logits: list[torch.Tensor | None] = [None] * self.batch_size
        self.logical_pos = [0] * self.batch_size
        self.target_ptr = [0] * self.batch_size
        self.kd_by_row: list[list[list[torch.Tensor]] | None] = [None] * self.batch_size
        self.kd_teacher_col_by_row: list[list[int]] = [
            [] for _ in range(self.batch_size)
        ]
        self.lm_sum = self.embed.weight.sum() * 0.0
        self.lm_count = torch.zeros((), device=self.device, dtype=torch.long)
        self.calls = 0
        self.tokens = 0

    def _token_embed(self, token_id: int) -> torch.Tensor:
        return self.embed(torch.tensor([token_id], device=self.device))[0]

    def _call_model(
        self,
        step_embeds: torch.Tensor,
        full_mask: torch.Tensor,
        pos_ids: torch.Tensor,
        cache,
        hidden_request: HiddenRequest,
        compute_logits: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, tuple | None, object]:
        kw = dict(
            inputs_embeds=step_embeds,
            attention_mask=full_mask,
            position_ids=pos_ids,
            past_key_values=cache,
            use_cache=True,
        )
        # Call decoder directly so lm_head can be skipped when logits are not needed.
        causal_lm = self.model.student.base_model.model
        want_hidden = hidden_request == "all_layers"
        out = causal_lm.model(**kw, output_hidden_states=want_hidden)
        last_hidden = out.last_hidden_state if hidden_request == "last_layer" else None
        hidden_states = out.hidden_states if want_hidden else None
        logits = causal_lm.lm_head(out.last_hidden_state) if compute_logits else None
        return logits, last_hidden, hidden_states, out.past_key_values

    @staticmethod
    def _rebuild_cache(chunks: tuple[tuple[torch.Tensor, ...], ...]) -> DynamicCache:
        if not chunks:
            return DynamicCache()
        num_layers = len(chunks[0]) // 2
        pairs = [
            (
                torch.cat([c[2 * l] for c in chunks], dim=2),
                torch.cat([c[2 * l + 1] for c in chunks], dim=2),
            )
            for l in range(num_layers)
        ]
        return DynamicCache(ddp_cache_data=pairs)

    def _forward_rebuild(
        self,
        step_embeds: torch.Tensor,
        full_mask: torch.Tensor,
        pos_ids: torch.Tensor,
        chunks: tuple[tuple[torch.Tensor, ...], ...],
        hidden_request: HiddenRequest,
        compute_logits: bool,
        prefix_len: int,
    ):
        """Rebuild cache from chunks each forward so backward recompute doesn't double-append."""
        cache = self._rebuild_cache(chunks)
        logits, last_hidden, hidden_states, new_cache = self._call_model(
            step_embeds, full_mask, pos_ids, cache, hidden_request, compute_logits
        )
        tail = tuple(
            t[:, :, prefix_len:, :].contiguous()
            for layer in new_cache.layers
            for t in (layer.keys, layer.values)
        )
        return logits, last_hidden, hidden_states, tail

    def _student_step(
        self,
        step_embeds: torch.Tensor,
        step_mask: torch.Tensor,
        hidden_request: HiddenRequest,
        *,
        compute_logits: bool = True,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, tuple | None]:
        self.calls += 1
        pos = torch.tensor(self.logical_pos, device=self.device)
        pos_ids = pos[:, None] + torch.arange(step_embeds.shape[1], device=self.device)
        full_mask = torch.cat([self.cache_mask, step_mask], dim=1)

        # Snapshot chunks: live list mutated below; checkpoint holds args by ref.
        args = (
            step_embeds,
            full_mask,
            pos_ids,
            tuple(self._kv_chunks),
            hidden_request,
            compute_logits,
            self.cache_mask.shape[1],
        )
        # Checkpoint every call to avoid O(n^2) cache growth on long batches; under
        # no_grad checkpoint runs the forward directly without recompute.
        logits, last_hidden, hidden_states, tail = checkpoint(
            self._forward_rebuild, *args, use_reentrant=False
        )
        self._kv_chunks.append(tail)
        self.cache_mask = full_mask
        return logits, last_hidden, hidden_states

    def _add_lm(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        if logits.shape[0] == 0:
            return
        self.lm_sum = self.lm_sum + F.cross_entropy(
            logits, labels, ignore_index=self.cfg.ignore_index, reduction="sum"
        )
        self.lm_count += (labels != self.cfg.ignore_index).sum()

    def _kd_offsets(self, row: int, span: _TextSpan) -> list[int]:
        offsets: list[int] = []
        targets = self.targets_by_row[row]
        while (
            self.target_ptr[row] < len(targets)
            and targets[self.target_ptr[row]] < span.start
        ):
            self.target_ptr[row] += 1
        while (
            self.target_ptr[row] < len(targets)
            and targets[self.target_ptr[row]] < span.end
        ):
            offsets.append(targets[self.target_ptr[row]] - span.start)
            self.target_ptr[row] += 1
        return offsets

    def _collect_kd(self, row: int, offsets: list[int], hidden_states: tuple) -> None:
        layers = select_kd_layers(hidden_states, self.cfg.kd_layers)
        if self.kd_by_row[row] is None:
            self.kd_by_row[row] = [[] for _ in layers]
        idx = torch.tensor(offsets, device=hidden_states[0].device)
        for layer_kd, hidden in zip(self.kd_by_row[row], layers, strict=True):
            layer_kd.extend(hidden[row, idx])
        ptr = self.target_ptr[row]
        self.kd_teacher_col_by_row[row].extend(
            self.targets_by_row[row][ptr - len(offsets) : ptr]
        )

    def _needs_hidden(self, spans: list[_TextSpan | None]) -> bool:
        for b, span in enumerate(spans):
            if span is None:
                continue
            targets = self.targets_by_row[b]
            ptr = self.target_ptr[b]
            while ptr < len(targets) and targets[ptr] < span.start:
                ptr += 1
            if ptr < len(targets) and targets[ptr] < span.end:
                return True
        return False

    def _span_inputs(
        self, spans: list[_TextSpan | None], width: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        step_embeds = self.embed.weight.new_zeros(self.batch_size, width, self.hidden)
        step_mask = torch.zeros(
            self.batch_size, width, device=self.device, dtype=torch.long
        )
        for b, span in enumerate(spans):
            if span is None:
                continue
            step_embeds[b, : span.length] = self.embed(
                self.input_ids[b, span.start : span.end]
            )
            step_mask[b, : span.length] = 1
        return step_embeds, step_mask

    def _consume_spans(
        self,
        spans: list[_TextSpan | None],
        logits: torch.Tensor,
        hidden_states: tuple | None,
    ) -> None:
        for b, span in enumerate(spans):
            if span is None:
                continue
            lab = self.labels[b, span.start : span.end]
            if self.prev_logits[b] is not None:
                self._add_lm(self.prev_logits[b][None], lab[:1])
            self._add_lm(logits[b, : span.length - 1], lab[1:])
            self.prev_logits[b] = logits[b, span.length - 1]

            offsets = self._kd_offsets(b, span)
            if offsets:
                if hidden_states is None:
                    raise RuntimeError("KD target requested hidden states, got none.")
                self._collect_kd(b, offsets, hidden_states)
            self.logical_pos[b] += span.length

    def _latent_call_count(self) -> int:
        return (
            int(self.latent_start_embed is not None)
            + self.cfg.latent_steps
            + int(self.latent_end_embed is not None)
        )

    def _latent_input_from_base(
        self, base: torch.Tensor, row_idx: torch.Tensor
    ) -> torch.Tensor:
        latent = base
        if self.model.thought_projector is not None:
            latent = self.model.thought_projector(
                base.to(device=self.proj_device, dtype=self.proj_dtype)
            )
        latent_in = self.embed.weight.new_zeros(self.batch_size, 1, self.hidden)
        latent_in[row_idx, 0] = latent.to(device=self.device, dtype=latent_in.dtype)
        return latent_in

    def _latent_block_forward(
        self,
        chunks: tuple[tuple[torch.Tensor, ...], ...],
        prefix_mask: torch.Tensor,
        base_pos: torch.Tensor,
        step_mask: torch.Tensor,
        latent_start_embeds: torch.Tensor,
        latent_end_embeds: torch.Tensor,
        row_idx: torch.Tensor,
    ):
        cache = self._rebuild_cache(chunks)
        running_mask = prefix_mask
        tails: list[torch.Tensor] = []
        offset = 0

        def one_call(embeds, hidden_request, compute_logits):
            nonlocal cache, running_mask, offset
            full_mask = torch.cat([running_mask, step_mask], dim=1)
            pos_ids = (base_pos + offset)[:, None]
            logits, last_hidden, _, cache = self._call_model(
                embeds, full_mask, pos_ids, cache, hidden_request, compute_logits
            )
            running_mask = full_mask
            offset += 1
            for layer in cache.layers:  # each call appends exactly one token
                tails.append(layer.keys[:, :, -1:, :].contiguous())
                tails.append(layer.values[:, :, -1:, :].contiguous())
            return logits, last_hidden

        final_logits = None
        base = self.embed.weight.new_zeros(row_idx.shape[0], self.hidden)
        if self.latent_start_embed is not None:
            is_final = self.cfg.latent_steps == 0 and self.latent_end_embed is None
            final_logits, last_hidden = one_call(
                latent_start_embeds,
                "none" if is_final else "last_layer",
                is_final,
            )
            if last_hidden is not None:
                base = last_hidden[row_idx, 0]

        for i in range(self.cfg.latent_steps):
            is_final = i + 1 == self.cfg.latent_steps and self.latent_end_embed is None
            final_logits, last_hidden = one_call(
                self._latent_input_from_base(base, row_idx),
                "none" if is_final else "last_layer",
                is_final,
            )
            if last_hidden is not None:
                base = last_hidden[row_idx, 0]

        if self.latent_end_embed is not None:
            final_logits, _ = one_call(latent_end_embeds, "none", True)

        return final_logits, tuple(tails)

    def _latent_block(self, rows: list[int]) -> None:
        if not rows or self._latent_call_count() == 0:
            return
        row_idx = torch.tensor(rows, device=self.device)
        step_mask = torch.zeros(
            self.batch_size, 1, device=self.device, dtype=torch.long
        )
        step_mask[row_idx, 0] = 1
        base_pos = torch.tensor(self.logical_pos, device=self.device)
        latent_start_embeds = self.embed.weight.new_zeros(
            self.batch_size, 1, self.hidden
        )
        if self.latent_start_embed is not None:
            latent_start_embeds[row_idx, 0] = self.latent_start_embed
        latent_end_embeds = self.embed.weight.new_zeros(self.batch_size, 1, self.hidden)
        if self.latent_end_embed is not None:
            latent_end_embeds[row_idx, 0] = self.latent_end_embed

        final_logits, tails = checkpoint(
            self._latent_block_forward,
            tuple(self._kv_chunks),
            self.cache_mask,
            base_pos,
            step_mask,
            latent_start_embeds,
            latent_end_embeds,
            row_idx,
            use_reentrant=False,
        )

        num_calls = self._latent_call_count()
        per = len(tails) // num_calls
        for c in range(num_calls):
            self._kv_chunks.append(tails[c * per : (c + 1) * per])
        self.cache_mask = torch.cat(
            [self.cache_mask, step_mask.repeat(1, num_calls)], dim=1
        )

        for b in rows:
            if self.prev_logits[b] is not None:
                self._add_lm(self.prev_logits[b][None], self.ignore_lab)
            self.prev_logits[b] = final_logits[b, 0]
            self.logical_pos[b] += num_calls
        self.calls += num_calls
        self.tokens += len(rows) * num_calls

    def run(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, KdVecs | None, int, int]:
        max_spans = max((len(c) for c in self.spans_by_row), default=0)
        for chunk_idx in range(max_spans):
            spans = [
                c[chunk_idx] if chunk_idx < len(c) else None for c in self.spans_by_row
            ]
            width = max((0 if s is None else s.length for s in spans), default=0)
            if width == 0:
                break

            step_embeds, step_mask = self._span_inputs(spans, width)
            hidden_request: HiddenRequest = (
                "all_layers" if self._needs_hidden(spans) else "none"
            )
            logits, _, hidden_states = self._student_step(
                step_embeds, step_mask, hidden_request
            )
            self.tokens += sum(s.length for s in spans if s is not None)
            self._consume_spans(spans, logits, hidden_states)

            latent_rows = [
                b
                for b, s in enumerate(spans)
                if s is not None and s.insert_latent_after
            ]
            self._latent_block(latent_rows)

        return self._finalize()

    def _finalize(self) -> tuple[torch.Tensor, torch.Tensor, KdVecs | None, int, int]:
        if any(k is not None for k in self.kd_by_row):
            num_layers = len(next(k for k in self.kd_by_row if k is not None))
            vecs = [
                torch.stack(
                    [
                        vec
                        for b in range(self.batch_size)
                        if self.kd_by_row[b] is not None
                        for vec in self.kd_by_row[b][layer]
                    ]
                )
                for layer in range(num_layers)
            ]
            row_col = torch.tensor(
                [
                    (b, col)
                    for b in range(self.batch_size)
                    for col in self.kd_teacher_col_by_row[b]
                ],
                dtype=self._pos_dtype,
                device=self.device,
            ).reshape(-1, 2)
            student_kd = KdVecs(row_col=row_col, vecs=vecs)
        else:
            student_kd = None

        return (
            self.lm_sum,
            self.lm_count,
            student_kd,
            self.calls,
            self.tokens,
        )


def streaming_student_outputs(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    batch_idx: torch.Tensor,
    teacher_pos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, KdVecs | None, int, int]:
    return _StreamingStudent(
        model,
        input_ids,
        labels,
        attention_mask,
        batch_idx,
        teacher_pos,
    ).run()
