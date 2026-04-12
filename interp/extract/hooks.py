"""Activation capture and steering hook infrastructure for CWM.

We monkey-patch cwm.fastgen.forward._forward with _hooked_forward(), which has
the same semantics but reads from module-global context variables to capture
activations and apply steering vectors.

Thread safety: _current_store is written by the worker thread (via
activation_hook_context) before g.generate(), which blocks until the forward
pass completes on the main thread.  Python's GIL makes simple assignments
atomic; ImpGen's blocking semantics supply the necessary happens-before.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, Shard

with suppress(ModuleNotFoundError, ImportError):
    from xformers.ops.fmha import merge_attentions
    from xformers.ops.fmha.attn_bias import (
        PagedBlockDiagonalCausalWithOffsetPaddedKeysMask as AttnBias,
    )
    from xformers.ops.fmha.flash3 import mha_fwd
    from xformers.ops.rmsnorm import rms_norm

with suppress(ModuleNotFoundError, ImportError):
    from flash_mla.flash_mla_interface import (
        flash_mla_with_kvcache,
        get_mla_metadata,
    )

with suppress(ModuleNotFoundError, ImportError):
    from cwm.fastgen.forward import (
        MlaPrefill,
        maybe_dist_to_local,
        vocab_parallel_embedding,
    )
    from cwm.fastgen.kernels import apply_rope, paged_memcpy


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@dataclass
class ActivationStore:
    """Accumulates hidden-state captures across decode steps.

    Multi-turn support: call next_turn() before each g.generate() call to
    increment current_turn.  Every activation captured during that call will
    have its turn index recorded in turn_indices (parallel to positions).
    """

    layers: list[int]
    capture_token_ids: list[int] | None  # None = capture all positions
    data: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    positions: list[int] = field(default_factory=list)
    # Parallel to positions: which turn (g.generate() call) each capture is from.
    turn_indices: list[int] = field(default_factory=list)
    # Incremented by next_turn(); read by _capture_at_layer().
    current_turn: int = 0
    enabled: bool = True
    # Sequence-position range for this trajectory's current generate call.
    # Set to (len(traj.context), len(traj.context) + max_gen) before each
    # g.generate() so that when the TP batch contains multiple trajectories
    # from different ranks, we only capture positions that belong to *this*
    # trajectory and not the TP partner's.
    seqlen_start: int | None = None
    seqlen_end: int | None = None

    def next_turn(self) -> None:
        """Advance the turn counter.  Call once before each g.generate() call."""
        self.current_turn += 1

    def clear(self) -> None:
        self.data.clear()
        self.positions.clear()
        self.turn_indices.clear()
        self.current_turn = 0
        self.seqlen_start = None
        self.seqlen_end = None

    def get_activations(self, layer: int) -> torch.Tensor | None:
        """Return [n_captured, dim] tensor for a layer, or None."""
        if layer not in self.data or not self.data[layer]:
            return None
        return torch.cat(self.data[layer], dim=0)


@dataclass
class SteeringHook:
    """Adds alpha * vector to h at a given layer."""

    layer: int
    vector: torch.Tensor  # [dim], will be moved to device as needed
    alpha: float = 1.0
    position: str = "all"  # all | last | trace_tokens


# ---------------------------------------------------------------------------
# Global hook context
# ---------------------------------------------------------------------------

_current_store: ActivationStore | None = None
_current_steering_hooks: list[SteeringHook] = []


class activation_hook_context:
    """Context manager that installs an ActivationStore and SteeringHooks."""

    def __init__(
        self,
        store: ActivationStore | None = None,
        steering_hooks: list[SteeringHook] | None = None,
    ) -> None:
        self.store = store
        self.steering_hooks = steering_hooks or []

    def __enter__(self) -> "activation_hook_context":
        global _current_store, _current_steering_hooks
        _current_store = self.store
        _current_steering_hooks = self.steering_hooks
        return self

    def __exit__(self, *args: object) -> None:
        global _current_store, _current_steering_hooks
        _current_store = None
        _current_steering_hooks = []


# ---------------------------------------------------------------------------
# Monkey-patch install / uninstall
# ---------------------------------------------------------------------------

_original_forward = None


def install_forward_hooks() -> None:
    """Replace cwm.fastgen.forward._forward with the hook-aware version."""
    global _original_forward
    import cwm.fastgen.forward as _fwd

    if _original_forward is None:
        _original_forward = _fwd._forward
    _fwd._forward = _hooked_forward


def uninstall_forward_hooks() -> None:
    """Restore the original cwm.fastgen.forward._forward."""
    global _original_forward
    if _original_forward is not None:
        import cwm.fastgen.forward as _fwd

        _fwd._forward = _original_forward
        _original_forward = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _capture_at_layer(
    store: ActivationStore,
    layer_idx: int,
    token_values: torch.Tensor,
    h: torch.Tensor,
    q_seqpos: torch.Tensor,
) -> None:
    if store.capture_token_ids is not None:
        cap_ids = torch.tensor(
            store.capture_token_ids, dtype=token_values.dtype, device=h.device
        )
        mask = torch.isin(token_values, cap_ids)
        if not mask.any():
            return
        captured = h[mask].clone().detach().cpu()
        indices = mask.nonzero(as_tuple=True)[0]
    else:
        captured = h.clone().detach().cpu()
        indices = torch.arange(h.shape[0], device=h.device)

    if layer_idx not in store.data:
        store.data[layer_idx] = []
    store.data[layer_idx].append(captured)

    # Record sequence positions and the current turn index
    n_captured = captured.shape[0]
    if q_seqpos.numel() > 1:
        pos_vals = q_seqpos[indices].tolist()
    else:
        pos_vals = [int(q_seqpos.item())] * n_captured
    store.positions.extend(pos_vals)
    store.turn_indices.extend([store.current_turn] * n_captured)


def _apply_steering_at_layer(
    hooks: list[SteeringHook],
    layer_idx: int,
    token_values: torch.Tensor,
    h: torch.Tensor,
    capture_ids: list[int] | None,
) -> None:
    for hook in hooks:
        if hook.layer != layer_idx or hook.alpha == 0.0:
            continue
        vec = hook.vector.to(device=h.device, dtype=h.dtype)
        if hook.position == "all":
            h.add_(hook.alpha * vec)
        elif hook.position == "last":
            h[-1:].add_(hook.alpha * vec)
        elif hook.position == "trace_tokens" and capture_ids is not None:
            cap_t = torch.tensor(capture_ids, dtype=token_values.dtype, device=h.device)
            mask = torch.isin(token_values, cap_t)
            if mask.any():
                h[mask].add_(hook.alpha * vec)


# ---------------------------------------------------------------------------
# Hook-aware forward pass (copy of _forward with hook points inserted)
# ---------------------------------------------------------------------------


@torch.inference_mode()
def _hooked_forward(
    model,
    coll,
    q_seqlen,
    actual_batch_size,
    token_values: torch.Tensor,
    attn_bias,
    mla_attn,
    cache,
    cache_shard: tuple[int, int],
    logits_idx,
    prefill: bool = True,
) -> torch.Tensor:
    """Hook-aware copy of cwm.fastgen.forward._forward.

    Identical to _forward except that after each layer's FFN residual addition
    (h.add_(h_out)) it:
      1. Captures h into _current_store (TP rank 0 only).
      2. Applies any _current_steering_hooks (all TP ranks).
    """
    store = _current_store
    steering = _current_steering_hooks

    qk_head_dim = model.qk_head_dim
    head_dim = model.head_dim
    nope_dim = model.qk_nope_head_dim
    rope_dim = model.qk_rope_head_dim
    n_layers = model.n_layers
    n_local_heads = model.layers[0].attention.n_heads
    n_local_kv_heads = model.layers[0].attention.n_kv_heads
    assert model.norm is not None, "unsupported model"
    eps = model.norm.eps

    cache.page_in(0)

    q_batch: torch.Tensor | None
    q_seqpos: torch.Tensor
    cache_len = attn_bias.k_seqinfo.seqlen

    if prefill:
        assert q_seqlen is not None
        q_batch = torch.tensor(
            sum(([i] * n for i, n in enumerate(q_seqlen)), []),
            dtype=torch.int,
            device=token_values.device,
        )
        k_seqlen = attn_bias.k_seqinfo.seqlen_py
        q_seqpos_list: list[int] = []
        for n, t in zip(k_seqlen, q_seqlen, strict=False):
            q_seqpos_list.extend(range(n - t, n))
        q_seqpos = torch.tensor(
            q_seqpos_list,
            dtype=torch.int,
            device=token_values.device,
        )
    else:
        q_batch = None
        q_seqpos = attn_bias.k_seqinfo.seqlen - 1
        if cache_shard[1] > 1:
            idx, cnt = cache_shard
            cache_len = (cache_len + cnt - 1 - idx) // cnt

    tok_embedding_weight = maybe_dist_to_local(model.tok_embeddings.weight)
    placement = (
        model.tok_embeddings.weight.placements[0]
        if isinstance(model.tok_embeddings.weight, DTensor)
        else None
    )
    if isinstance(placement, Shard) and placement.dim == 0:
        assert coll is not None
        h = vocab_parallel_embedding(tok_embedding_weight, token_values, coll)
    else:
        h_parallel = F.embedding(token_values, tok_embedding_weight)
        if coll is not None:
            h = coll.all_gather(h_parallel)
        else:
            h = h_parallel

    if model.kv_lora_rank > 0 and not prefill:
        n_heads = n_local_heads
        if coll is not None:
            n_heads *= coll.tp_size
        tile_scheduler_metadata, num_splits = get_mla_metadata(
            cache_seqlens=cache_len,
            num_heads_per_head_k=n_heads,
            num_heads_k=1,
        )

    # TP rank-0 check used for capture only (steering applies on all ranks)
    is_tp_rank_zero = coll is None or coll.rank == 0

    for i, layer in enumerate(model.layers):
        if i + 1 < n_layers:
            cache.page_in(i + 1)

        attention_norm_weight = maybe_dist_to_local(layer.attention_norm.weight)
        h_in_attn = rms_norm(h, attention_norm_weight, eps)

        if model.qkv_biases:
            wq_bias = maybe_dist_to_local(layer.attention.wq.bias)
            wk_bias = maybe_dist_to_local(layer.attention.wk.bias)
            wv_bias = maybe_dist_to_local(layer.attention.wv.bias)
        else:
            wq_bias, wk_bias, wv_bias = None, None, None

        attn = layer.attention

        if attn.q_lora_rank == 0:
            wq_weight = maybe_dist_to_local(attn.wq.weight)
        else:
            wq_weight = maybe_dist_to_local(attn.wq_b.weight)

        if attn.kv_lora_rank > 0 and not prefill:
            wq_weight = attn._wq_full

        if attn.q_lora_rank == 0:
            xq = F.linear(h_in_attn, wq_weight, wq_bias)
        else:
            wqa_weight = maybe_dist_to_local(attn.wq_a.weight)
            qnorm_weight = maybe_dist_to_local(attn.q_norm.weight)
            xq_lora = F.linear(h_in_attn, wqa_weight)
            xq_lora = rms_norm(xq_lora, qnorm_weight, eps)
            xq = F.linear(xq_lora, wq_weight)

        if attn.kv_lora_rank == 0:
            wk_weight = maybe_dist_to_local(attn.wk.weight)
            wv_weight = maybe_dist_to_local(attn.wv.weight)
            xk = F.linear(h_in_attn, wk_weight, wk_bias)
            xv = F.linear(h_in_attn, wv_weight, wv_bias)

            xq = xq.view(xq.shape[0], n_local_heads, qk_head_dim)
            xk = xk.view(xk.shape[0], n_local_kv_heads, qk_head_dim)
            xv = xv.view(xv.shape[0], n_local_kv_heads, head_dim)

            if model.qk_norm:
                k_norm_weight = maybe_dist_to_local(layer.attention.k_norm.weight)
                q_norm_weight = maybe_dist_to_local(layer.attention.q_norm.weight)
                xk = rms_norm(xk, k_norm_weight, eps)
                xq = rms_norm(xq, q_norm_weight, eps)

            apply_rope(xq, q_seqpos, model.rope_freqs)
            apply_rope(xk, q_seqpos, model.rope_freqs)

            cache_k, cache_v = cache.cache_kv(i)
            for x, c in (xk, cache_k), (xv, cache_v):
                paged_memcpy(
                    src=x.flatten(1, 2),
                    dst=c,
                    page_tbl=attn_bias.block_tables,
                    dst_pos=q_seqpos,
                    dst_shard=None,
                    src_batch=q_batch,
                    batch_size=actual_batch_size,
                    page_size=attn_bias.page_size,
                )

            cache.page_out(i)

            kv_shape = (-1, attn_bias.page_size, n_local_kv_heads)
            attn_out, _ = mha_fwd(
                query=xq,
                key=cache_k.reshape(*kv_shape, qk_head_dim),
                value=cache_v.reshape(*kv_shape, head_dim),
                cu_seqlens_q=attn_bias.q_seqinfo.seqstart,
                cu_seqlens_k=None,
                seqused_k=attn_bias.k_seqinfo.seqlen,
                leftpad_k=None,
                max_seqlen_q=attn_bias.q_seqinfo.max_seqlen,
                max_seqlen_k=attn_bias.k_seqinfo.max_seqlen,
                p=0,
                block_table=attn_bias.block_tables,
                softmax_scale=xq.shape[-1] ** -0.5,
                is_causal=True,
                window_left=layer.window_size_left,
                window_right=-1,
            )
            attn_wo_weight = maybe_dist_to_local(attn.wo.weight)

        else:
            # MLA path
            assert not model.qk_norm, "mla + qk_norm unsupported"
            assert not model.qkv_biases, "mla + qkv_biases unsupported"

            lora_dim = attn.kv_lora_rank
            wkva_weight = maybe_dist_to_local(attn.wkv_a.weight)
            kvnorm_weight = maybe_dist_to_local(attn.kv_norm.weight)
            wkvb_weight = maybe_dist_to_local(attn.wkv_b.weight)
            n_heads = n_local_heads
            if not prefill and coll is not None:
                n_heads *= coll.tp_size

            xkv = F.linear(h_in_attn, wkva_weight)
            xkv, xk_pe = torch.split(xkv, [lora_dim, rope_dim], dim=-1)
            xk_pe = xk_pe.unsqueeze(1).contiguous()
            apply_rope(xk_pe, q_seqpos, model.rope_freqs)

            xkv = xkv.contiguous()
            xkv = rms_norm(xkv, kvnorm_weight, eps)

            (cache_xkv,) = cache.cache_kv(i)
            paged_memcpy(
                src=torch.cat([xkv, xk_pe.squeeze(1)], dim=-1),
                dst=cache_xkv,
                page_tbl=attn_bias.block_tables,
                dst_pos=q_seqpos,
                dst_shard=cache_shard,
                src_batch=q_batch,
                batch_size=actual_batch_size,
                page_size=attn_bias.page_size,
            )

            cache.page_out(i)

            xq = xq.unflatten(-1, (n_heads, -1))
            xq_nope, xq_pe = torch.split(xq, [nope_dim, rope_dim], dim=-1)
            xq_pe = xq_pe.contiguous()
            apply_rope(xq_pe, q_seqpos, model.rope_freqs)

            if prefill:
                assert mla_attn is not None
                attn_out = mla_attn.run(
                    xq=torch.cat([xq_nope, xq_pe], dim=-1),
                    xkv=xkv,
                    xk_pe=xk_pe,
                    cache=cache_xkv,
                    wkvb=wkvb_weight,
                    head_dim=head_dim,
                    nope_dim=nope_dim,
                    rope_dim=rope_dim,
                    lora_dim=lora_dim,
                    n_kv_heads=n_heads,
                    coll=coll,
                )
                attn_wo_weight = maybe_dist_to_local(attn.wo.weight)
            else:
                xq_nope = xq_nope.transpose(0, 1)
                xq_nope = torch.bmm(xq_nope, attn._wkb)
                xq_nope = xq_nope.transpose(0, 1)
                xq = torch.cat([xq_nope, xq_pe], dim=-1)

                cache_xkv = cache_xkv.reshape(
                    -1,
                    attn_bias.page_size,
                    1,
                    lora_dim + rope_dim,
                )
                attn_out, attn_lse = flash_mla_with_kvcache(
                    q=xq.unsqueeze(1),
                    k_cache=cache_xkv,
                    block_table=attn_bias.block_tables,
                    cache_seqlens=cache_len,
                    head_dim_v=lora_dim,
                    tile_scheduler_metadata=tile_scheduler_metadata,
                    num_splits=num_splits,
                    softmax_scale=(nope_dim + rope_dim) ** -0.5,
                )

                if coll is not None:
                    attn_out, _ = merge_attentions(
                        coll.all_gather(attn_out[None], 0),
                        coll.all_gather(attn_lse[None], 0),
                        write_lse=False,
                    )

                attn_out = attn_out.squeeze(1).transpose(0, 1)
                attn_out = torch.bmm(attn_out, attn._wv)
                attn_out = attn_out.transpose(0, 1)
                attn_wo_weight = attn._wo_full

        # attn residual
        h_out = F.linear(attn_out.flatten(1, 2), attn_wo_weight)
        if (attn.kv_lora_rank == 0 or prefill) and coll is not None:
            h_out = coll.all_reduce(h_out)
        h.add_(h_out)

        # FFN
        ffn_norm_weight = maybe_dist_to_local(layer.ffn_norm.weight)
        h_in_ffn = rms_norm(h, ffn_norm_weight, eps)

        feed_forward_w1_weight = maybe_dist_to_local(layer.feed_forward.w1.weight)
        feed_forward_w3_weight = maybe_dist_to_local(layer.feed_forward.w3.weight)
        x1 = F.linear(h_in_ffn, feed_forward_w1_weight)
        x3 = F.linear(h_in_ffn, feed_forward_w3_weight)

        feed_forward_w2_weight = maybe_dist_to_local(layer.feed_forward.w2.weight)
        x_mul = F.silu(x1, inplace=True).mul_(x3)
        h_out = F.linear(x_mul, feed_forward_w2_weight)

        if coll is not None:
            h_out = coll.all_reduce(h_out)
        h.add_(h_out)

        # -------------------------------------------------------------------
        # HOOK POINTS
        # -------------------------------------------------------------------
        # Capture: rank 0 only (h is identical across ranks after all-reduce)
        if store is not None and store.enabled and i in store.layers and is_tp_rank_zero:
            _capture_at_layer(store, i, token_values, h, q_seqpos)

        # Steering: all ranks (modification must be consistent across TP)
        if steering:
            _apply_steering_at_layer(
                steering, i, token_values, h, store.capture_token_ids if store else None
            )

    # Final norm + output head (unchanged)
    norm_weight = maybe_dist_to_local(model.norm.weight)
    h = rms_norm(h, norm_weight, eps)
    if logits_idx is not None:
        assert prefill
        h = h[logits_idx]
    if hasattr(model.output, "weight") and model.output.weight is not None:
        original_output_weight = model.output.weight
    elif hasattr(model.output, "tied_module") and model.output.tied_module is not None:
        original_output_weight = model.output.tied_module.weight
    else:
        raise AttributeError(
            "model.output has neither 'weight' nor 'tied_module' attribute"
        )
    output_weight = maybe_dist_to_local(original_output_weight)
    if isinstance(original_output_weight, DTensor):
        assert coll is not None
        placement = original_output_weight.placements[0]
        assert isinstance(placement, Shard), f"{placement}"
        if placement.dim == 0:
            logits_parallel = F.linear(h, output_weight)
            logits = coll.all_gather(logits_parallel)
        elif placement.dim == 1:
            h = coll.local_split(h)
            logits_parallel = F.linear(h, output_weight)
            logits = coll.all_reduce(logits_parallel)
        else:
            raise ValueError(f"unexpected placement dim: {placement.dim}")
    else:
        logits = F.linear(h, output_weight)
    return logits.float()
