# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Teacher-forcing CODI distillation on CRUXEval-O execution traces.

Loads a HuggingFace CWM checkpoint as a LoRA student, builds the ground-truth
trace dataset in memory (``cwm.training.data``), and optimizes the combined LM
+ hidden-state KD loss from ``CodiModel``. The teacher is the same base model
with adapters disabled, avoiding a second model copy. The student is
teacher-forced (ground-truth embeddings fed at every step).

    python -m cwm.training.train model_name_or_path=facebook/cwm output_dir=./codi-cruxeval
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from cwm.training.codi import CodiModel, default_codi_config_from_tokenizer
from cwm.training.data import IGNORE_INDEX, build_dataset

logger = logging.getLogger(__name__)


@dataclass
class TrainArgs:
    model_name_or_path: str = "facebook/cwm"
    output_dir: str = "./codi-cruxeval"
    latent_steps: int = 6
    use_thought_projector: bool = True
    lr: float = 1e-4
    epochs: int = 1
    batch_size: int = 1
    grad_accum_steps: int = 32
    max_grad_norm: float = 1.0
    save_every_steps: int = 0  # 0 = save only at end
    n_samples: int = -1
    max_seq_len: int = 8192
    seed: int = 42
    wandb_log: bool = False
    wandb_name: str = ""
    wandb_log_every: int = 1
    timing_log_every: int = 0  # 0 = disabled; log rank0 timing every N optimizer steps
    device_map: str = "auto"
    num_workers: int = 16
    flash_attn: bool = True   # sdpa fused-kernel attention (no extra package needed)
    tp_size: int = 1
    dp_size: int = 0  # 0 = infer from WORLD_SIZE / (tp_size * pp_size)
    pp_size: int = 1  # reserved config entry; pipeline parallel is not wired yet
    tp_plan: str = "auto"


@dataclass
class _DistState:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    dp_size: int
    tp_size: int
    pp_size: int
    dp_rank: int
    tp_rank: int
    pp_rank: int
    dp_group: Any | None = None
    device_mesh: DeviceMesh | None = None

    @property
    def is_rank_zero(self) -> bool:
        return self.rank == 0

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def dp_enabled(self) -> bool:
        return self.dp_size > 1

    @property
    def tp_enabled(self) -> bool:
        return self.tp_size > 1

    @property
    def is_dp_leader(self) -> bool:
        return self.tp_rank == 0 and self.pp_rank == 0


def _init_distributed(args: TrainArgs) -> "_DistState":
    if args.tp_size <= 0:
        raise ValueError("tp_size must be positive")
    if args.dp_size < 0:
        raise ValueError("dp_size must be non-negative")
    if args.pp_size <= 0:
        raise ValueError("pp_size must be positive")
    if args.pp_size != 1:
        raise NotImplementedError(
            "pp_size is reserved for future pipeline parallel support; use pp_size=1."
        )

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    else:
        rank = 0
        local_rank = 0
    parallel_group_size = args.tp_size * args.pp_size
    if world_size % parallel_group_size != 0:
        raise ValueError(
            "WORLD_SIZE must be divisible by tp_size * pp_size: "
            f"{world_size=} {args.tp_size=} {args.pp_size=}"
        )
    inferred_dp_size = world_size // parallel_group_size
    dp_size = args.dp_size or inferred_dp_size
    if dp_size != inferred_dp_size:
        raise ValueError(
            "dp_size must match WORLD_SIZE / (tp_size * pp_size): "
            f"{dp_size=} {world_size=} {args.tp_size=} {args.pp_size=}"
        )
    if args.tp_size > 1 and world_size == 1:
        raise ValueError("tp_size > 1 requires launching with torchrun/srun tasks.")

    pp_rank = rank // (dp_size * args.tp_size)
    rank_in_pp = rank % (dp_size * args.tp_size)
    dp_rank = rank_in_pp // args.tp_size
    tp_rank = rank_in_pp % args.tp_size
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    device_mesh = None
    dp_group = None
    if world_size > 1:
        device_mesh = init_device_mesh(
            "cuda",
            (dp_size, args.tp_size),
            mesh_dim_names=("dp", "tp"),
        )
        if dp_size > 1:
            dp_group = device_mesh["dp"].get_group()

    return _DistState(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        dp_size=dp_size,
        tp_size=args.tp_size,
        pp_size=args.pp_size,
        dp_rank=dp_rank,
        tp_rank=tp_rank,
        pp_rank=pp_rank,
        dp_group=dp_group,
        device_mesh=device_mesh,
    )


def _destroy_distributed(state: "_DistState") -> None:
    if state.enabled:
        dist.destroy_process_group()



def _sync_gradients(params: list[torch.nn.Parameter], state: _DistState) -> None:
    if not state.dp_enabled:
        return
    for param in params:
        if param.grad is None:
            continue
        if param.grad.device.type != "cuda":
            raise RuntimeError(
                f"Cannot synchronize non-CUDA gradient on {param.grad.device}"
            )
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=state.dp_group)
        param.grad.div_(state.dp_size)


def _collate(batch, pad_id: int) -> dict[str, torch.Tensor]:
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


def main(args: TrainArgs) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dist_state = _init_distributed(args)
    try:
        torch.manual_seed(args.seed + dist_state.dp_rank)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

        # Trace format: the token after <|action_sep|> is source (not <|eot_id|>),
        # so KD positions must not require an eot next token.
        config = default_codi_config_from_tokenizer(
            tokenizer, latent_steps=args.latent_steps, require_action_next_eot=False
        )

        load_device_map = args.device_map

        def load():
            kwargs: dict = {"dtype": torch.bfloat16}
            if args.flash_attn:
                kwargs["attn_implementation"] = "sdpa"
            if dist_state.tp_enabled:
                if args.tp_plan in (None, "none", "None"):
                    raise ValueError("tp_size > 1 requires tp_plan, usually tp_plan=auto")
                kwargs["tp_plan"] = args.tp_plan
                kwargs["device_mesh"] = dist_state.device_mesh
            elif load_device_map not in (None, "none", "None"):
                kwargs["device_map"] = load_device_map
            return AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path, **kwargs
            )

        student = load()
        model = CodiModel(
            student=student,
            config=config,
            use_thought_projector=args.use_thought_projector,
        )
        is_sharded_model = dist_state.tp_enabled or load_device_map not in (None, "none", "None")
        if not is_sharded_model:
            model.to(dist_state.device)
        model.train()

        # Gradient checkpointing is intentionally omitted: the step-by-step
        # KV-cache forward in CodiModel requires use_cache=True. LoRA weights get
        # gradients without forcing frozen embedding outputs to require grad; the
        # latter only bloats the long streaming autograd graph.

        pad_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        device = model.input_embeddings.weight.device

        # CRUXEval trace rendering executes the dataset's own Python programs, whose
        # set/dict iteration order depends on each process's random PYTHONHASHSEED.
        # Under pure TP every rank would otherwise build a subtly different dataset
        # (different rendered values -> different token lengths), so the per-layer TP
        # all_reduce hits mismatched shapes ([1,79,6144] vs [1,71,6144]) and aborts.
        # Build once on the global leader and broadcast byte-identical data to all
        # ranks; with the seeded DataLoader generator below the whole TP group stays
        # in lockstep on both data content and shuffle order.
        if dist_state.is_rank_zero or not dist_state.enabled:
            dataset = build_dataset(
                tokenizer, n_samples=args.n_samples, max_seq_len=args.max_seq_len
            )
        else:
            dataset = None
        if dist_state.enabled:
            obj = [dataset]
            dist.broadcast_object_list(obj, src=0, device=dist_state.device)
            dataset = obj[0]
        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=dist_state.dp_size,
                rank=dist_state.dp_rank,
                shuffle=True,
                seed=args.seed,
            )
            if dist_state.dp_enabled
            else None
        )
        # Pure TP (dp_size=1) gives every rank no sampler, so DataLoader falls back
        # to its own RandomSampler. Without a shared generator each rank shuffles
        # independently and feeds a different (variable-length) sequence, so the TP
        # all_reduce sees mismatched shapes across ranks and aborts. A generator
        # seeded identically on every rank keeps all TP ranks in lockstep order.
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            collate_fn=lambda b: _collate(b, pad_id),
            num_workers=args.num_workers,
            pin_memory=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
        if args.grad_accum_steps <= 0:
            raise ValueError("grad_accum_steps must be positive")

        params = [p for p in model.parameters() if p.requires_grad]
        logger.info(
            "rank %d/%d dp=%d/%d tp=%d/%d pp=%d/%d: "
            "%d examples, %d local batches, %d trainable params",
            dist_state.rank,
            dist_state.world_size,
            dist_state.dp_rank,
            dist_state.dp_size,
            dist_state.tp_rank,
            dist_state.tp_size,
            dist_state.pp_rank,
            dist_state.pp_size,
            len(dataset),
            len(loader),
            sum(p.numel() for p in params),
        )
        optimizer = torch.optim.AdamW(params, lr=args.lr)

        use_wandb = args.wandb_log and dist_state.is_rank_zero
        if use_wandb:
            os.environ.setdefault("WANDB_MODE", "offline")
            os.environ.setdefault("WANDB_DIR", args.output_dir)
            import wandb

            wandb.init(
                project="codi_distill_cwm",
                name=args.wandb_name or None,
                config=vars(args),
            )

        out_dir = Path(args.output_dir)

        def save_checkpoint(path: Path) -> None:
            if not dist_state.is_rank_zero:
                return
            if dist_state.tp_enabled:
                logger.warning(
                    "Saving LoRA adapter from TP rank 0 only; validate the "
                    "checkpoint before relying on TP-sharded adapter reloads."
                )
            path.mkdir(parents=True, exist_ok=True)
            model.student.save_pretrained(path)  # LoRA adapter only
            if model.thought_projector is not None:
                torch.save(
                    model.thought_projector.state_dict(),
                    path / "thought_projector.pt",
                )
            logger.info("Saved checkpoint to %s", path)

        step = 0

        profile_this_rank = args.timing_log_every > 0 and dist_state.is_rank_zero

        def timing_stamp() -> float:
            if profile_this_rank and dist_state.device.type == "cuda":
                torch.cuda.synchronize(dist_state.device)
            return time.perf_counter()

        def new_timing() -> dict:
            return {
                "data": 0.0,
                "h2d": 0.0,
                "forward": 0.0,
                "student": 0.0,
                "teacher": 0.0,
                "kd_compute": 0.0,
                "backward": 0.0,
                "optimizer": 0.0,
                "student_calls": 0.0,
                "student_tokens": 0.0,
            }

        def optimizer_step(
            epoch: int, window: dict, accum_batches: int, timing: dict
        ) -> None:
            nonlocal step
            opt_start = timing_stamp()
            # Partial final window: rescale grads so the average matches a full window.
            if accum_batches < args.grad_accum_steps:
                scale = args.grad_accum_steps / accum_batches
                for param in params:
                    if param.grad is not None:
                        param.grad.mul_(scale)
            _sync_gradients(params, dist_state)
            grad_norm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            timing["optimizer"] += timing_stamp() - opt_start
            step += 1

            metrics = torch.tensor(
                [
                    window["loss"],
                    window["lm"],
                    window["kd"],
                    window["kd_pos"],
                    float(accum_batches),
                ],
                device=device,
                dtype=torch.float32,
            )
            if dist_state.dp_enabled:
                dist.all_reduce(metrics, op=dist.ReduceOp.SUM, group=dist_state.dp_group)
            total_batches = float(metrics[4].item())
            loss = float(metrics[0].item()) / total_batches
            lm = float(metrics[1].item()) / total_batches
            kd = float(metrics[2].item()) / total_batches
            kd_pos = float(metrics[3].item()) / total_batches
            lr = optimizer.param_groups[0]["lr"]

            if dist_state.is_rank_zero:
                logger.info(
                    "epoch %d step %d loss=%.4f lm=%.4f kd=%.4f "
                    "kd_pos=%.1f grad_norm=%.3f",
                    epoch,
                    step,
                    loss,
                    lm,
                    kd,
                    kd_pos,
                    float(grad_norm),
                )
                if use_wandb and step % args.wandb_log_every == 0:
                    wandb.log(
                        {
                            "loss": loss,
                            "lm_loss": lm,
                            "kd_loss": kd,
                            "num_kd_positions": kd_pos,
                            "grad_norm": float(grad_norm),
                            "lr": lr,
                            "epoch": epoch,
                        },
                        step=step,
                    )
            if profile_this_rank and step % args.timing_log_every == 0:
                total = (
                    timing["data"]
                    + timing["h2d"]
                    + timing["forward"]
                    + timing["backward"]
                    + timing["optimizer"]
                )
                logger.info(
                    "timing step %d total=%.3fs data=%.3f h2d=%.3f "
                    "forward=%.3f student=%.3f teacher=%.3f kd_compute=%.3f "
                    "backward=%.3f optimizer=%.3f student_calls=%.0f "
                    "student_tokens=%.0f accum_batches=%d",
                    step,
                    total,
                    timing["data"],
                    timing["h2d"],
                    timing["forward"],
                    timing["student"],
                    timing["teacher"],
                    timing["kd_compute"],
                    timing["backward"],
                    timing["optimizer"],
                    timing["student_calls"],
                    timing["student_tokens"],
                    accum_batches,
                )
            if args.save_every_steps and step % args.save_every_steps == 0:
                save_checkpoint(out_dir / f"checkpoint-{step}")

        def new_window() -> dict:
            return {"loss": 0.0, "lm": 0.0, "kd": 0.0, "kd_pos": 0.0}

        optimizer.zero_grad()
        for epoch in range(args.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            accum_batches = 0
            window = new_window()
            timing = new_timing()
            num_batches = len(loader)
            loader_iter = iter(loader)
            for batch_idx in range(num_batches):
                start = timing_stamp()
                batch = next(loader_iter)
                timing["data"] += timing_stamp() - start

                start = timing_stamp()
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                timing["h2d"] += timing_stamp() - start

                will_step = (
                    accum_batches + 1 == args.grad_accum_steps
                    or batch_idx + 1 == num_batches
                )

                start = timing_stamp()
                out = model(
                    batch["input_ids"],
                    labels=batch["labels"],
                    attention_mask=batch["attention_mask"],
                    profile=profile_this_rank,
                )
                timing["forward"] += timing_stamp() - start
                if profile_this_rank:
                    timing["student"] += out.metrics["time_student_s"].item()
                    timing["teacher"] += out.metrics["time_teacher_s"].item()
                    timing["kd_compute"] += out.metrics["time_kd_s"].item()
                    timing["student_calls"] += out.metrics["student_model_calls"].item()
                    timing["student_tokens"] += out.metrics["student_tokens"].item()

                start = timing_stamp()
                (out.loss / args.grad_accum_steps).backward()
                timing["backward"] += timing_stamp() - start

                accum_batches += 1
                window["loss"] += out.loss.item()
                window["lm"] += out.lm_loss.item()
                window["kd"] += out.kd_loss.item()
                window["kd_pos"] += out.metrics["num_kd_positions"].item()

                if will_step:
                    optimizer_step(epoch, window, accum_batches, timing)
                    accum_batches = 0
                    window = new_window()
                    timing = new_timing()

        if torch.cuda.is_available():
            peak_alloc = torch.cuda.max_memory_allocated(dist_state.device) / 1024**3
            peak_reserved = torch.cuda.max_memory_reserved(dist_state.device) / 1024**3
            logger.info(
                "rank %d/%d peak VRAM: %.1f GB allocated / %.1f GB reserved",
                dist_state.rank, dist_state.world_size, peak_alloc, peak_reserved,
            )

    finally:
        _destroy_distributed(dist_state)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from cwm.common.params import load_from_cli

    main(load_from_cli(TrainArgs))
