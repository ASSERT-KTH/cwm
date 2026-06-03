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

import torch
import torch.distributed as dist

from cwm.training.codi import CodiModel
from cwm.training.codi_config import default_codi_config_from_tokenizer
from cwm.training.data import build_codi_dataloader
from cwm.training.distributed import (
    destroy_distributed,
    init_distributed,
    sync_gradients,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainArgs:
    model_name_or_path: str = "facebook/cwm"
    output_dir: str = "./codi-cruxeval"
    latent_steps: int = 2
    kd_layers: list[int] | None = None  # None = all layers; e.g. [-1] for last only
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
    dp_size: int = 0  # 0 = infer from WORLD_SIZE / tp_size
    tp_plan: str = "auto"


def main(args: TrainArgs) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dist_state = init_distributed(tp_size=args.tp_size, dp_size=args.dp_size)
    try:
        torch.manual_seed(args.seed + dist_state.dp_rank)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

        config = default_codi_config_from_tokenizer(
            tokenizer,
            latent_steps=args.latent_steps,
            kd_layers=tuple(args.kd_layers) if args.kd_layers else None,
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

        pad_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        device = model.input_embeddings.weight.device

        loader, sampler = build_codi_dataloader(
            tokenizer=tokenizer,
            pad_id=pad_id,
            dist_state=dist_state,
            n_samples=args.n_samples,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
        )
        if args.grad_accum_steps <= 0:
            raise ValueError("grad_accum_steps must be positive")

        params = [p for p in model.parameters() if p.requires_grad]
        logger.info(
            "rank %d/%d dp=%d/%d tp=%d/%d: "
            "%d examples, %d local batches, %d trainable params",
            dist_state.rank,
            dist_state.world_size,
            dist_state.dp_rank,
            dist_state.dp_size,
            dist_state.tp_rank,
            dist_state.tp_size,
            len(loader.dataset),
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

        collect_timing = args.timing_log_every > 0 and dist_state.is_rank_zero

        def timing_stamp() -> float:
            if collect_timing and dist_state.device.type == "cuda":
                torch.cuda.synchronize(dist_state.device)
            return time.perf_counter()

        def new_timing() -> dict:
            return {
                "data": 0.0,
                "h2d": 0.0,
                "forward": 0.0,
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
            sync_gradients(params, dist_state)
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
            if collect_timing and step % args.timing_log_every == 0:
                total = (
                    timing["data"]
                    + timing["h2d"]
                    + timing["forward"]
                    + timing["backward"]
                    + timing["optimizer"]
                )
                logger.info(
                    "timing step %d total=%.3fs data=%.3f h2d=%.3f "
                    "forward=%.3f backward=%.3f optimizer=%.3f student_calls=%.0f "
                    "student_tokens=%.0f accum_batches=%d",
                    step,
                    total,
                    timing["data"],
                    timing["h2d"],
                    timing["forward"],
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
                )
                timing["forward"] += timing_stamp() - start
                if collect_timing:
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
        destroy_distributed(dist_state)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from cwm.common.params import load_from_cli

    main(load_from_cli(TrainArgs))
