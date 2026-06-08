# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Compare CODI teacher/student forward order CUDA memory.

Runs one or more training batches without optimizer updates and reports CUDA
allocated/reserved peaks for either the old student-first order or the new
teacher-first order. Launch with torchrun to mirror DP ranks; each rank logs its
own memory because the rank-local batch can differ.
"""

from __future__ import annotations

import gc
import logging
import os
import time
from dataclasses import dataclass, field
import torch

from cwm.training.codi import CodiModel
from cwm.training.codi_config import default_codi_config_from_tokenizer
from cwm.training.codi_streaming import streaming_student_outputs
from cwm.training.data import build_codi_dataloader
from cwm.training.distributed import destroy_distributed, init_distributed


logger = logging.getLogger(__name__)


@dataclass
class ProbeArgs:
    model_name_or_path: str = "model_weights/cwm_hf"
    order: str = "teacher_first"
    qlora: bool = True
    latent_steps: int = 1
    kd_layers: list[int] | None = field(default_factory=lambda: [-1])
    use_thought_projector: bool = True
    lm_loss_weight: float = 1.0
    kd_loss_weight: float = 1.0
    n_samples: int = 0
    data_split: str = "train"
    max_seq_len: int = 1536
    max_batch_tokens: int = 1536
    batch_size: int = 32
    grad_accum_steps: int = 1
    num_workers: int = 0
    megabatch_mult: int = 8
    seed: int = 42
    num_batches: int = 1
    skip_batches: int = 0
    run_backward: bool = True
    empty_cache_between_batches: bool = True
    flash_attn: bool = True
    tp_size: int = 1
    dp_size: int = 0
    log_all_ranks: bool = True


def _gb(nbytes: int) -> float:
    return nbytes / 1024**3


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _cleanup(device: torch.device, *, empty_cache: bool) -> None:
    gc.collect()
    if device.type == "cuda":
        _sync(device)
        if empty_cache:
            torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _mem(device: torch.device) -> str:
    if device.type != "cuda":
        return "cpu"
    _sync(device)
    return (
        f"alloc={_gb(torch.cuda.memory_allocated(device)):.2f}GB "
        f"reserved={_gb(torch.cuda.memory_reserved(device)):.2f}GB "
        f"peak_alloc={_gb(torch.cuda.max_memory_allocated(device)):.2f}GB "
        f"peak_reserved={_gb(torch.cuda.max_memory_reserved(device)):.2f}GB"
    )


def _teacher_forward(
    model: CodiModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    batch_idx: torch.Tensor,
    teacher_pos: torch.Tensor,
):
    with torch.no_grad():
        disable_adapter = getattr(model.student, "disable_adapter", None)
        if disable_adapter is None:
            raise RuntimeError("CodiModel needs a PEFT student with disable_adapter().")
        was_training = model.student.training
        model.student.eval()
        try:
            with disable_adapter():
                return model._teacher_kd_vecs(
                    input_ids,
                    attention_mask,
                    batch_idx,
                    teacher_pos,
                )
        finally:
            model.student.train(was_training)


def _student_forward(
    model: CodiModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    batch_idx: torch.Tensor,
    teacher_pos: torch.Tensor,
):
    return streaming_student_outputs(
        model,
        input_ids,
        labels,
        attention_mask,
        batch_idx,
        teacher_pos,
    )


def _loss_from_parts(
    model: CodiModel,
    student_parts,
    teacher_kd_vecs,
):
    lm_sum, lm_count, student_kd_vecs, _student_pos, calls, tokens = student_parts
    lm_loss = lm_sum / lm_count.clamp(min=1)
    kd_loss = model._kd_loss(student_kd_vecs, teacher_kd_vecs, lm_loss)
    loss = model.config.lm_loss_weight * lm_loss + model.config.kd_loss_weight * kd_loss
    return loss, lm_loss.detach(), kd_loss.detach(), calls, tokens


def _run_one_batch(args: ProbeArgs, model: CodiModel, batch, device, dist_state, batch_id: int) -> None:
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    attention_mask = batch["attention_mask"]
    batch_idx, teacher_pos = model._teacher_positions(input_ids, labels, attention_mask)

    should_log = args.log_all_ranks or dist_state.is_rank_zero
    if should_log:
        logger.info(
            "order=%s rank=%d batch=%d shape=%s kd_pos=%d start %s",
            args.order,
            dist_state.rank,
            batch_id,
            tuple(input_ids.shape),
            batch_idx.numel(),
            _mem(device),
        )

    _reset_peak(device)
    start = time.perf_counter()
    if args.order == "teacher_first":
        teacher_kd_vecs = _teacher_forward(model, input_ids, attention_mask, batch_idx, teacher_pos)
        if should_log:
            logger.info(
                "order=%s rank=%d batch=%d after_teacher %.3fs %s",
                args.order,
                dist_state.rank,
                batch_id,
                time.perf_counter() - start,
                _mem(device),
            )
        student_parts = _student_forward(model, input_ids, labels, attention_mask, batch_idx, teacher_pos)
        if should_log:
            logger.info(
                "order=%s rank=%d batch=%d after_student %.3fs %s",
                args.order,
                dist_state.rank,
                batch_id,
                time.perf_counter() - start,
                _mem(device),
            )
    else:
        student_parts = _student_forward(model, input_ids, labels, attention_mask, batch_idx, teacher_pos)
        if should_log:
            logger.info(
                "order=%s rank=%d batch=%d after_student %.3fs %s",
                args.order,
                dist_state.rank,
                batch_id,
                time.perf_counter() - start,
                _mem(device),
            )
        teacher_kd_vecs = _teacher_forward(model, input_ids, attention_mask, batch_idx, teacher_pos)
        if should_log:
            logger.info(
                "order=%s rank=%d batch=%d after_teacher %.3fs %s",
                args.order,
                dist_state.rank,
                batch_id,
                time.perf_counter() - start,
                _mem(device),
            )

    loss, lm_loss, kd_loss, calls, tokens = _loss_from_parts(model, student_parts, teacher_kd_vecs)
    if should_log:
        logger.info(
            "order=%s rank=%d batch=%d loss_built loss=%.4f lm=%.4f kd=%.4f calls=%d tokens=%d %s",
            args.order,
            dist_state.rank,
            batch_id,
            float(loss.detach()),
            float(lm_loss),
            float(kd_loss),
            calls,
            tokens,
            _mem(device),
        )

    if args.run_backward:
        loss.backward()
        if should_log:
            logger.info(
                "order=%s rank=%d batch=%d after_backward %.3fs %s",
                args.order,
                dist_state.rank,
                batch_id,
                time.perf_counter() - start,
                _mem(device),
            )
        model.zero_grad(set_to_none=True)

    del loss, lm_loss, kd_loss, student_parts, teacher_kd_vecs, batch
    _cleanup(device, empty_cache=args.empty_cache_between_batches)
    if should_log:
        logger.info(
            "order=%s rank=%d batch=%d after_cleanup %s",
            args.order,
            dist_state.rank,
            batch_id,
            _mem(device),
        )


def main(args: ProbeArgs) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.order not in {"student_first", "teacher_first"}:
        raise ValueError(f"order must be student_first or teacher_first, got {args.order!r}")
    if args.qlora and args.tp_size > 1:
        raise ValueError("qlora (bitsandbytes 4-bit) is incompatible with tp_size > 1")

    dist_state = init_distributed(tp_size=args.tp_size, dp_size=args.dp_size)
    try:
        torch.manual_seed(args.seed + dist_state.dp_rank)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        kd_layers = tuple(args.kd_layers) if args.kd_layers else None
        config = default_codi_config_from_tokenizer(
            tokenizer,
            latent_steps=args.latent_steps,
            kd_layers=kd_layers,
            lm_loss_weight=args.lm_loss_weight,
            kd_loss_weight=args.kd_loss_weight,
        )

        kwargs: dict = {"dtype": torch.bfloat16}
        if args.flash_attn:
            kwargs["attn_implementation"] = "sdpa"
        if args.qlora:
            from transformers import BitsAndBytesConfig

            kwargs["device_map"] = {"": dist_state.local_rank}
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        if dist_state.tp_enabled:
            kwargs["tp_plan"] = "auto"
            kwargs["device_mesh"] = dist_state.device_mesh

        student = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **kwargs)
        model = CodiModel(
            student=student,
            config=config,
            use_thought_projector=args.use_thought_projector,
        )
        if not (dist_state.tp_enabled or args.qlora):
            model.to(dist_state.device)
        model.train()
        device = model.input_embeddings.weight.device

        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        loader, sampler = build_codi_dataloader(
            tokenizer=tokenizer,
            pad_id=pad_id,
            dist_state=dist_state,
            n_samples=args.n_samples,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps,
            num_workers=args.num_workers,
            seed=args.seed,
            megabatch_mult=args.megabatch_mult,
            max_batch_tokens=args.max_batch_tokens,
            split=args.data_split,
        )

        for batch_id, batch in enumerate(loader):
            if batch_id < args.skip_batches:
                continue
            if batch_id >= args.skip_batches + args.num_batches:
                break
            _run_one_batch(args, model, batch, device, dist_state, batch_id)

    finally:
        destroy_distributed(dist_state)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from cwm.common.params import load_from_cli

    main(load_from_cli(ProbeArgs))
