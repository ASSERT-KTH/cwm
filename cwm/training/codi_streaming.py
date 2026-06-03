# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.cache_utils import DynamicCache

HiddenRequest = Literal["none", "all_layers", "last_layer"]


@dataclass(frozen=True)
class _ChunkSpan:
    start: int
    end: int
    has_line_sep: bool

    @property
    def length(self) -> int:
        return self.end - self.start


def _chunk_spans(line_sep_id: int, row: list[int], valid_len: int) -> list[_ChunkSpan]:
    """Split one row into spans, each ending at a <|line_sep|> (or the row end)."""
    chunks, prev = [], 0
    for i, tok in enumerate(row[:valid_len]):
        if tok == line_sep_id:
            chunks.append(_ChunkSpan(prev, i + 1, True))
            prev = i + 1
    if prev < valid_len:
        chunks.append(_ChunkSpan(prev, valid_len, False))
    return chunks


class _StreamingStudent:
    """One teacher-forced streaming pass over a batch of CRUXEval traces.

    Text chunks (split at ``<|line_sep|>``) and the injected latent tokens are
    fed to the LoRA student incrementally through a KV cache: LM loss is summed
    per token and student hidden states are collected at the KD (teacher)
    positions. The latent embedding at each step is produced from the previous
    step's last hidden state, so the pass is inherently sequential.

    With ``gradient_checkpointing`` every student call recomputes its activations
    during backward instead of storing them, which removes the dominant memory
    cost of the long streaming graph. The recompute must be pure, so instead of
    mutating one shared KV cache (which would double-append on recompute) each
    call rebuilds a fresh cache by concatenating the per-call key/value chunks of
    all earlier calls. Those chunks are kept **with gradient** (only the recomputable
    per-layer activations are dropped), so the full cross-call cache gradient --
    and hence the latent generation gradient, which reaches the loss only through
    later tokens attending to the injected latents -- is preserved exactly.
    """

    def __init__(
        self,
        model,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        batch_idx: torch.Tensor,
        teacher_pos: torch.Tensor,
        *,
        gradient_checkpointing: bool,
    ) -> None:
        self.model = model
        self.cfg = model.config
        self.input_ids = input_ids
        self.labels = labels
        self.device = input_ids.device
        self.embed = model.input_embeddings
        self.hidden = self.embed.weight.shape[-1]
        self.gradient_checkpointing = gradient_checkpointing
        self.batch_size = input_ids.shape[0]
        self._pos_dtype = batch_idx.dtype

        self.sot_embed = self._token_embed(self.cfg.sot_token_id)
        self.eot_embed = self._token_embed(self.cfg.eot_token_id)
        self.ignore_lab = labels.new_full((1,), self.cfg.ignore_index)

        rows = input_ids.tolist()
        valid_lens = attention_mask.sum(dim=1).tolist()
        self.chunks_by_row = [
            _chunk_spans(self.cfg.line_sep_token_id, rows[b], valid_lens[b])
            for b in range(self.batch_size)
        ]
        self.targets_by_row = [
            teacher_pos[batch_idx == b].tolist() for b in range(self.batch_size)
        ]

        self.proj_device = self.proj_dtype = None
        if model.thought_projector is not None:
            param = next(model.thought_projector.parameters())
            self.proj_device, self.proj_dtype = param.device, param.dtype

        # Streaming state. Only one cache representation is live at a time:
        # ``_cache`` for the plain path; ``_kv_chunks`` (one grad-carrying flat
        # (k0, v0, k1, v1, ...) tuple of *new* per-layer key/values per call) when
        # checkpointing rebuilds a fresh cache for each recomputable call.
        self._cache = None
        self._kv_chunks: list[tuple[torch.Tensor, ...]] = []
        self.cache_mask = torch.zeros(self.batch_size, 0, device=self.device, dtype=torch.long)
        self.prev_logits: list[torch.Tensor | None] = [None] * self.batch_size
        self.logical_pos = [0] * self.batch_size
        self.target_ptr = [0] * self.batch_size
        self.kd_by_row: list[list[list[torch.Tensor]] | None] = [None] * self.batch_size
        self.kd_pos_by_row: list[list[int]] = [[] for _ in range(self.batch_size)]
        self.lm_sum = self.embed.weight.sum() * 0.0
        self.lm_count = torch.zeros((), device=self.device, dtype=torch.long)
        self.calls = 0
        self.tokens = 0

    def _token_embed(self, token_id: int) -> torch.Tensor:
        return self.embed(torch.tensor([token_id], device=self.device))[0]

    # -- student forward (the checkpoint boundary) --------------------------

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
        causal_lm = self.model._causal_lm_module()
        if hidden_request == "last_layer" and causal_lm is not None:
            out = causal_lm.model(**kw)
            logits = causal_lm.lm_head(out.last_hidden_state) if compute_logits else None
            return logits, out.last_hidden_state, None, out.past_key_values
        if hidden_request == "last_layer":
            out = self.model.student(**kw, output_hidden_states=True)
            logits = out.logits if compute_logits else None
            return logits, out.hidden_states[-1], out.hidden_states, out.past_key_values
        out = self.model.student(**kw, output_hidden_states=hidden_request == "all_layers")
        return out.logits, None, out.hidden_states, out.past_key_values

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
        """Checkpointable: rebuilds a fresh cache by concatenating the earlier
        per-call key/value chunks (so the backward recompute does not double-append
        to a shared cache), runs the student, and returns only this call's new
        key/value tail to extend the chunk list with."""
        if chunks:
            num_layers = len(chunks[0]) // 2
            pairs = [
                (
                    torch.cat([chunk[2 * l] for chunk in chunks], dim=2),
                    torch.cat([chunk[2 * l + 1] for chunk in chunks], dim=2),
                )
                for l in range(num_layers)
            ]
            cache = DynamicCache(ddp_cache_data=pairs)
        else:
            cache = DynamicCache()
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

        if self.gradient_checkpointing and torch.is_grad_enabled():
            # Snapshot the current chunks (the live list is mutated below, and the
            # checkpoint holds its args by reference until the backward recompute).
            args = (
                step_embeds,
                full_mask,
                pos_ids,
                tuple(self._kv_chunks),
                hidden_request,
                compute_logits,
                self.cache_mask.shape[1],
            )
            # Every call -- including the single-token latent calls -- must be
            # checkpointed. Each call rebuilds (or, on the plain path, grows) a
            # cache by concatenating all prior key/values; that growing O(L) cat is
            # retained in the autograd graph unless it lives inside a checkpoint, so
            # leaving any call eager re-accumulates an O(n^2) cache and OOMs on long
            # batches. Recompute drops it and rebuilds inside the backward pass.
            logits, last_hidden, hidden_states, tail = checkpoint(
                self._forward_rebuild, *args, use_reentrant=False
            )
            self._kv_chunks.append(tail)
        else:
            logits, last_hidden, hidden_states, self._cache = self._call_model(
                step_embeds, full_mask, pos_ids, self._cache, hidden_request, compute_logits
            )

        self.cache_mask = full_mask
        return logits, last_hidden, hidden_states

    # -- loss / kd accumulation ---------------------------------------------

    def _add_lm(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        if logits.shape[0] == 0:
            return
        self.lm_sum = self.lm_sum + F.cross_entropy(
            logits, labels, ignore_index=self.cfg.ignore_index, reduction="sum"
        )
        self.lm_count += (labels != self.cfg.ignore_index).sum()

    def _kd_offsets(self, row: int, span: _ChunkSpan) -> list[int]:
        offsets: list[int] = []
        targets = self.targets_by_row[row]
        while self.target_ptr[row] < len(targets) and targets[self.target_ptr[row]] < span.end:
            target = targets[self.target_ptr[row]]
            if target >= span.start:
                offsets.append(target - span.start)
            self.target_ptr[row] += 1
        return offsets

    def _collect_kd(
        self, row: int, offsets: list[int], start_pos: int, hidden_states: tuple
    ) -> None:
        if self.kd_by_row[row] is None:
            self.kd_by_row[row] = [[] for _ in hidden_states[1:]]
        idx = torch.tensor(offsets, device=hidden_states[0].device)
        for layer_kd, hidden in zip(self.kd_by_row[row], hidden_states[1:], strict=True):
            layer_kd.extend(hidden[row, idx])
        self.kd_pos_by_row[row].extend(start_pos + off for off in offsets)

    # -- chunk / latent steps -----------------------------------------------

    def _needs_hidden(self, spans: list[_ChunkSpan | None]) -> bool:
        return any(
            span is not None
            and self.target_ptr[b] < len(self.targets_by_row[b])
            and self.targets_by_row[b][self.target_ptr[b]] < span.end
            for b, span in enumerate(spans)
        )

    def _chunk_inputs(
        self, spans: list[_ChunkSpan | None], width: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        step_embeds = self.embed.weight.new_zeros(self.batch_size, width, self.hidden)
        step_mask = torch.zeros(self.batch_size, width, device=self.device, dtype=torch.long)
        for b, span in enumerate(spans):
            if span is None:
                continue
            step_embeds[b, : span.length] = self.embed(self.input_ids[b, span.start : span.end])
            step_mask[b, : span.length] = 1
        return step_embeds, step_mask

    def _consume_chunk(
        self, spans: list[_ChunkSpan | None], logits: torch.Tensor, hidden_states: tuple | None
    ) -> None:
        for b, span in enumerate(spans):
            if span is None:
                continue
            lab = self.labels[b, span.start : span.end]
            start_pos = self.logical_pos[b]
            if self.prev_logits[b] is not None:
                self._add_lm(self.prev_logits[b][None], lab[:1])
            self._add_lm(logits[b, : span.length - 1], lab[1:])
            self.prev_logits[b] = logits[b, span.length - 1]

            offsets = self._kd_offsets(b, span)
            if offsets:
                if hidden_states is None:
                    raise RuntimeError("KD target requested hidden states, got none.")
                self._collect_kd(b, offsets, start_pos, hidden_states)
            self.logical_pos[b] += span.length

    def _latent_step(
        self,
        step_embeds: torch.Tensor,
        rows: list[int],
        hidden_request: Literal["none", "last_layer"],
        *,
        compute_logits: bool,
    ) -> torch.Tensor | None:
        mask = torch.zeros(self.batch_size, 1, device=self.device, dtype=torch.long)
        for b in rows:
            mask[b, 0] = 1
        logits, last_hidden, _ = self._student_step(
            step_embeds, mask, hidden_request, compute_logits=compute_logits
        )
        for b in rows:
            if self.prev_logits[b] is not None:
                self._add_lm(self.prev_logits[b][None], self.ignore_lab)
            self.prev_logits[b] = logits[b, 0] if logits is not None else None
            self.logical_pos[b] += 1
        self.tokens += len(rows)
        return last_hidden

    def _latent_block(self, rows: list[int]) -> None:
        if not rows:
            return
        row_idx = torch.tensor(rows, device=self.device)
        if self.gradient_checkpointing and torch.is_grad_enabled():
            self._latent_block_checkpointed(rows, row_idx)
        else:
            self._latent_block_plain(rows, row_idx)

    def _latent_block_plain(self, rows: list[int], row_idx: torch.Tensor) -> None:
        step_embeds = self.embed.weight.new_zeros(self.batch_size, 1, self.hidden)
        step_embeds[row_idx, 0] = self.sot_embed
        last_hidden = self._latent_step(step_embeds, rows, "last_layer", compute_logits=False)
        base = last_hidden[row_idx, 0]

        for _ in range(self.cfg.latent_steps):
            latent = base
            if self.model.thought_projector is not None:
                latent = self.model.thought_projector(
                    base.to(device=self.proj_device, dtype=self.proj_dtype)
                )
            latent_in = self.embed.weight.new_zeros(self.batch_size, 1, self.hidden)
            latent_in[row_idx, 0] = latent.to(device=self.device, dtype=latent_in.dtype)
            last_hidden = self._latent_step(latent_in, rows, "last_layer", compute_logits=False)
            base = last_hidden[row_idx, 0]

        step_embeds = self.embed.weight.new_zeros(self.batch_size, 1, self.hidden)
        step_embeds[row_idx, 0] = self.eot_embed
        self._latent_step(step_embeds, rows, "none", compute_logits=True)

    def _latent_block_forward(
        self,
        chunks: tuple[tuple[torch.Tensor, ...], ...],
        prefix_mask: torch.Tensor,
        base_pos: torch.Tensor,
        step_mask: torch.Tensor,
        sot_embeds: torch.Tensor,
        eot_embeds: torch.Tensor,
        row_idx: torch.Tensor,
    ):
        """Checkpointable: runs the whole latent block (sot + latent_steps + eot)
        as one recompute unit. Rebuilds the cache once from the earlier chunks,
        then grows it in place across the single-token internal calls, applying the
        thought projector between them. Returns the eot logits (for the next chunk's
        first-token LM handoff) and the flat new key/value tail of every internal
        call (one per token) to extend the chunk list with."""
        if chunks:
            num_layers = len(chunks[0]) // 2
            pairs = [
                (
                    torch.cat([c[2 * l] for c in chunks], dim=2),
                    torch.cat([c[2 * l + 1] for c in chunks], dim=2),
                )
                for l in range(num_layers)
            ]
            cache = DynamicCache(ddp_cache_data=pairs)
        else:
            cache = DynamicCache()

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

        _, last_hidden = one_call(sot_embeds, "last_layer", False)
        base = last_hidden[row_idx, 0]
        for _ in range(self.cfg.latent_steps):
            latent = base
            if self.model.thought_projector is not None:
                latent = self.model.thought_projector(
                    base.to(device=self.proj_device, dtype=self.proj_dtype)
                )
            latent_in = self.embed.weight.new_zeros(self.batch_size, 1, self.hidden)
            latent_in[row_idx, 0] = latent.to(device=self.device, dtype=latent_in.dtype)
            _, last_hidden = one_call(latent_in, "last_layer", False)
            base = last_hidden[row_idx, 0]
        eot_logits, _ = one_call(eot_embeds, "none", True)

        return eot_logits, tuple(tails)

    def _latent_block_checkpointed(self, rows: list[int], row_idx: torch.Tensor) -> None:
        step_mask = torch.zeros(self.batch_size, 1, device=self.device, dtype=torch.long)
        step_mask[row_idx, 0] = 1
        base_pos = torch.tensor(self.logical_pos, device=self.device)
        sot_embeds = self.embed.weight.new_zeros(self.batch_size, 1, self.hidden)
        sot_embeds[row_idx, 0] = self.sot_embed
        eot_embeds = self.embed.weight.new_zeros(self.batch_size, 1, self.hidden)
        eot_embeds[row_idx, 0] = self.eot_embed

        eot_logits, tails = checkpoint(
            self._latent_block_forward,
            tuple(self._kv_chunks),
            self.cache_mask,
            base_pos,
            step_mask,
            sot_embeds,
            eot_embeds,
            row_idx,
            use_reentrant=False,
        )

        # Split the flat tail back into one (k0, v0, k1, v1, ...) tuple per call.
        num_calls = self.cfg.latent_steps + 2
        per = len(tails) // num_calls
        for c in range(num_calls):
            self._kv_chunks.append(tails[c * per : (c + 1) * per])
        self.cache_mask = torch.cat([self.cache_mask, step_mask.repeat(1, num_calls)], dim=1)

        # Only live effect of the block: hand the eot logits to the next chunk's
        # first-token LM term (the per-step ignore-label adds are no-ops).
        for b in rows:
            if self.prev_logits[b] is not None:
                self._add_lm(self.prev_logits[b][None], self.ignore_lab)
            self.prev_logits[b] = eot_logits[b, 0]
            self.logical_pos[b] += num_calls
        self.calls += num_calls
        self.tokens += len(rows) * num_calls

    # -- driver --------------------------------------------------------------

    def run(self) -> tuple[torch.Tensor, list[torch.Tensor] | None, torch.Tensor, int, int]:
        max_chunks = max((len(c) for c in self.chunks_by_row), default=0)
        for chunk_idx in range(max_chunks):
            spans = [c[chunk_idx] if chunk_idx < len(c) else None for c in self.chunks_by_row]
            width = max((0 if s is None else s.length for s in spans), default=0)
            if width == 0:
                break

            step_embeds, step_mask = self._chunk_inputs(spans, width)
            hidden_request: HiddenRequest = "all_layers" if self._needs_hidden(spans) else "none"
            logits, _, hidden_states = self._student_step(step_embeds, step_mask, hidden_request)
            self.tokens += sum(s.length for s in spans if s is not None)
            self._consume_chunk(spans, logits, hidden_states)

            latent_rows = [
                b for b, s in enumerate(spans) if s is not None and s.has_line_sep
            ]
            self._latent_block(latent_rows)

        return self._finalize()

    def _finalize(self) -> tuple[torch.Tensor, list[torch.Tensor] | None, torch.Tensor, int, int]:
        if any(k is not None for k in self.kd_by_row):
            num_layers = len(next(k for k in self.kd_by_row if k is not None))
            stacked_kd = [
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
            student_pos = [pos for row in self.kd_pos_by_row for pos in row]
        else:
            stacked_kd = None
            student_pos = []

        lm_loss = self.lm_sum / self.lm_count.clamp(min=1)
        return (
            lm_loss,
            stacked_kd,
            torch.tensor(student_pos, device=self.device, dtype=self._pos_dtype),
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
    *,
    gradient_checkpointing: bool = False,
) -> tuple[torch.Tensor, list[torch.Tensor] | None, torch.Tensor, int, int]:
    return _StreamingStudent(
        model,
        input_ids,
        labels,
        attention_mask,
        batch_idx,
        teacher_pos,
        gradient_checkpointing=gradient_checkpointing,
    ).run()
