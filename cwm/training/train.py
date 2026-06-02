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
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
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
    wandb_log_every: int = 10  # optimizer steps per wandb log point
    device_map: str = "auto"
    num_workers: int = 16
    flash_attn: bool = True   # sdpa fused-kernel attention (no extra package needed)


@dataclass
class _DistState:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_rank_zero(self) -> bool:
        return self.rank == 0

    @property
    def enabled(self) -> bool:
        return self.world_size > 1


def _init_distributed() -> "_DistState":
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        local_rank = 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return _DistState(rank=rank, local_rank=local_rank, world_size=world_size, device=device)


def _destroy_distributed(state: "_DistState") -> None:
    if state.enabled:
        dist.destroy_process_group()



def _sync_gradients(params: list[torch.nn.Parameter], state: _DistState) -> None:
    if not state.enabled:
        return
    for param in params:
        if param.grad is None:
            continue
        if param.grad.device.type != "cuda":
            raise RuntimeError(
                f"Cannot synchronize non-CUDA gradient on {param.grad.device}"
            )
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(state.world_size)


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

    dist_state = _init_distributed()
    try:
        torch.manual_seed(args.seed + dist_state.rank)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

        # Trace format: the token after <|action_sep|> is source (not <|eot_id|>),
        # so KD positions must not require an eot next token.
        config = default_codi_config_from_tokenizer(
            tokenizer, latent_steps=args.latent_steps, require_action_next_eot=False
        )

        load_device_map = args.device_map

        def load():
            kwargs: dict = {"dtype": torch.bfloat16}
            if load_device_map not in (None, "none", "None"):
                kwargs["device_map"] = load_device_map
            if args.flash_attn:
                kwargs["attn_implementation"] = "sdpa"
            return AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path, **kwargs
            )

        student = load()
        model = CodiModel(
            student=student,
            config=config,
            use_thought_projector=args.use_thought_projector,
        )
        is_sharded_model = load_device_map not in (None, "none", "None")
        if not is_sharded_model:
            model.to(dist_state.device)
        model.train()

        # enable_input_require_grads lets gradients flow through the frozen base
        # embeddings to reach LoRA adapters. Gradient checkpointing is intentionally
        # omitted: the step-by-step KV-cache forward in CodiModel is incompatible
        # with GC (HF forces use_cache=False, which breaks past_key_values
        # accumulation and causes both correctness bugs and near-zero GPU utilization).
        model.student.enable_input_require_grads()

        pad_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        device = model.input_embeddings.weight.device

        dataset = build_dataset(
            tokenizer, n_samples=args.n_samples, max_seq_len=args.max_seq_len
        )
        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=dist_state.world_size,
                rank=dist_state.rank,
                shuffle=True,
                seed=args.seed,
            )
            if dist_state.enabled
            else None
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            collate_fn=lambda b: _collate(b, pad_id),
            num_workers=args.num_workers,
            pin_memory=True,
        )
        if args.grad_accum_steps <= 0:
            raise ValueError("grad_accum_steps must be positive")

        params = [p for p in model.parameters() if p.requires_grad]
        logger.info(
            "rank %d/%d: %d examples, %d local batches, %d trainable params",
            dist_state.rank,
            dist_state.world_size,
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
            path.mkdir(parents=True, exist_ok=True)
            model.student.save_pretrained(path)  # LoRA adapter only
            if model.thought_projector is not None:
                torch.save(
                    model.thought_projector.state_dict(),
                    path / "thought_projector.pt",
                )
            logger.info("Saved checkpoint to %s", path)

        step = 0

        def optimizer_step(epoch: int, window: dict, accum_batches: int) -> None:
            nonlocal step
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
            if dist_state.enabled:
                dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
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
            num_batches = len(loader)
            for batch_idx, batch in enumerate(loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                will_step = (
                    accum_batches + 1 == args.grad_accum_steps
                    or batch_idx + 1 == num_batches
                )
                out = model(
                    batch["input_ids"],
                    labels=batch["labels"],
                    attention_mask=batch["attention_mask"],
                )
                (out.loss / args.grad_accum_steps).backward()

                accum_batches += 1
                window["loss"] += out.loss.item()
                window["lm"] += out.lm_loss.item()
                window["kd"] += out.kd_loss.item()
                window["kd_pos"] += out.metrics["num_kd_positions"].item()

                if will_step:
                    optimizer_step(epoch, window, accum_batches)
                    accum_batches = 0
                    window = new_window()

        if torch.cuda.is_available():
            peak_alloc = torch.cuda.max_memory_allocated(dist_state.device) / 1024**3
            peak_reserved = torch.cuda.max_memory_reserved(dist_state.device) / 1024**3
            logger.info(
                "rank %d/%d peak VRAM: %.1f GB allocated / %.1f GB reserved",
                dist_state.rank, dist_state.world_size, peak_alloc, peak_reserved,
            )

        save_checkpoint(out_dir)
    finally:
        _destroy_distributed(dist_state)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from cwm.common.params import load_from_cli

    main(load_from_cli(TrainArgs))
