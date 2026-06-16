from __future__ import annotations

import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import bitsandbytes
import torch
import torch.distributed as dist
from omegaconf import MISSING

from cwm.training.codi import CodiModel
from cwm.training.codi_config import default_codi_config_from_tokenizer
from cwm.training.data import build_codi_dataloader
from cwm.training.distributed import (
    destroy_distributed,
    gather_lora_full_state_dict,
    init_distributed,
    sync_gradients,
)
import wandb

logger = logging.getLogger(__name__)


@dataclass
class TrainArgs:
    model_name_or_path: str = "facebook/cwm"
    output_dir: str = "./codi-cruxeval"
    latent_steps: int = 2
    kd_layers: list[int] | None = field(
        default_factory=lambda: [-1]
    )  # [-1] = last layer only; None for all layers
    use_thought_projector: bool = True
    qlora: bool = False  # base + teacher in 4-bit NF4 (bitsandbytes)
    lr: float = 1e-4
    teacher_loss_weight: float = 1.0  # alpha: explicit-CoT teacher CE
    lm_loss_weight: float = 1.0       # beta: implicit-CoT student CE
    kd_loss_weight: float = 1.0       # gamma: hidden-state distillation
    epochs: int = 1
    batch_size: int = 1
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    save_every_steps: int = 0  # 0 = save only at end
    n_samples: int = -1
    data_source: list[str] = MISSING  # required, e.g. data_source=[mbpp,humaneval]
    max_seq_len: int = 8192
    build_workers: int = 0  # 0/1 = serial dataset build; >1 = forked Pool (rank0 only)
    megabatch_mult: int = 8  # length-bucketing strength; larger = tighter packing
    max_batch_tokens: int = 0  # 0 = fixed batch_size; >0 = padded-token budget
    seed: int = 42
    wandb_log: bool = False
    wandb_name: str = ""
    timing_log_every: int = 0  # 0 = disabled; log rank0 timing every N optimizer steps
    num_workers: int = 16
    tp_size: int = 1
    dp_size: int = 0  # 0 = infer from WORLD_SIZE / tp_size
    tp_plan: str = "auto"


def _build_model(args: TrainArgs, dist_state, config) -> CodiModel:
    """Load the base student and wrap it in CodiModel, placed on this rank."""
    from transformers import AutoModelForCausalLM

    kwargs: dict = {"dtype": torch.bfloat16, "attn_implementation": "sdpa"}
    if args.qlora:
        from transformers import BitsAndBytesConfig

        # 4-bit weights can't .to(); load straight onto this rank's GPU.
        kwargs["device_map"] = {"": dist_state.local_rank}
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    if dist_state.tp_enabled:
        if args.tp_plan in (None, "none", "None"):
            raise ValueError("tp_size > 1 requires tp_plan, usually tp_plan=auto")
        kwargs["tp_plan"] = args.tp_plan
        kwargs["device_mesh"] = dist_state.device_mesh

    student = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **kwargs)
    model = CodiModel(
        student=student,
        config=config,
        use_thought_projector=args.use_thought_projector,
    )

    # TP/qlora are pre-sharded onto their device; others need a move.
    if not (dist_state.tp_enabled or args.qlora):
        model.to(dist_state.device)
    model.train()
    return model


def _setup_wandb(args: TrainArgs, dist_state):
    """Init wandb on rank0 and crash-finish the run on SIGTERM. None if disabled."""
    if not (args.wandb_log and dist_state.is_rank_zero):
        return None

    if args.wandb_name:
        name = args.wandb_name
    else:
        kd = "all" if args.kd_layers is None else "-".join(map(str, args.kd_layers))
        name = (
            f"codi_{'qlora' if args.qlora else 'full'}_lat{args.latent_steps}_kd{kd}"
            f"_bs{args.batch_size}x{args.grad_accum_steps}_lr{args.lr:g}"
            f"_tw{args.teacher_loss_weight:g}_lmw{args.lm_loss_weight:g}_kdw{args.kd_loss_weight:g}"
            f"_seq{args.max_seq_len}_mbt{args.max_batch_tokens}"
            f"_ep{args.epochs}_dp{dist_state.dp_size}_tp{dist_state.tp_size}"
            f"_proj{int(args.use_thought_projector)}_seed{args.seed}"
        )

    os.environ["WANDB_MODE"] = "online"
    os.environ["WANDB_DIR"] = args.output_dir

    wandb_run = wandb.init(
        project="codi_distill_cwm",
        name=name,
        config=vars(args),
    )

    def _on_sigterm(signum, frame):
        wandb_run.finish(exit_code=1)
        os._exit(1)

    signal.signal(signal.SIGTERM, _on_sigterm)
    return wandb_run


def _save_checkpoint(model: CodiModel, dist_state, path: Path) -> None:
    """Save the LoRA adapter (+ thought projector) from rank 0 only."""
    # Under TP the LoRA weights are DTensors sharded across the TP mesh; gather
    # them to full tensors first. This is a collective, so every rank must call
    # it before the rank-0 write guard below or NCCL deadlocks.
    state_dict = gather_lora_full_state_dict(model.student) if dist_state.tp_enabled else None
    if not dist_state.is_rank_zero:
        return
    path.mkdir(parents=True, exist_ok=True)
    model.student.save_pretrained(path, state_dict=state_dict)  # LoRA adapter only
    if model.thought_projector is not None:
        torch.save(model.thought_projector.state_dict(), path / "thought_projector.pt")
    logger.info("Saved checkpoint to %s", path)


_TIMING_KEYS = (
    "data",
    "h2d",
    "forward",
    "backward",
    "optimizer",
    "student_calls",
    "student_tokens",
    "teacher_peak",  # GB, max over teacher fwd+bwd
    "student_peak",  # GB, max over student fwd+bwd (teacher graph already freed)
)


def _new_window() -> dict:
    return {"loss": 0.0, "teacher": 0.0, "lm": 0.0, "kd": 0.0}


def _new_timing() -> dict:
    return {k: 0.0 for k in _TIMING_KEYS}


class _Trainer:
    """Owns the optimizer-step / epoch loop and its mutable counters."""

    def __init__(self, args, dist_state, model, params, optimizer, out_dir, wandb_run):
        self.args = args
        self.dist_state = dist_state
        self.model = model
        self.params = params
        self.optimizer = optimizer
        self.out_dir = out_dir
        self.wandb_run = wandb_run
        self.device = model.input_embeddings.weight.device
        self.step = 0
        self.collect_timing = args.timing_log_every > 0 and dist_state.is_rank_zero

    def _stamp(self) -> float:
        if self.collect_timing and self.dist_state.device.type == "cuda":
            torch.cuda.synchronize(self.dist_state.device)
        return time.perf_counter()

    def _save_checkpoint(self, path: Path) -> None:
        _save_checkpoint(self.model, self.dist_state, path)

    def _log_timing(self, timing: dict, accum_batches: int) -> None:
        total = (
            timing["data"]
            + timing["h2d"]
            + timing["forward"]
            + timing["backward"]
            + timing["optimizer"]
        )
        device = self.dist_state.device
        logger.info(
            "timing step %d total=%.3fs data=%.3f h2d=%.3f "
            "forward=%.3f backward=%.3f optimizer=%.3f student_calls=%.0f "
            "student_tokens=%.0f teacher_peak=%.1fGB student_peak=%.1fGB accum_batches=%d",
            self.step,
            total,
            timing["data"],
            timing["h2d"],
            timing["forward"],
            timing["backward"],
            timing["optimizer"],
            timing["student_calls"],
            timing["student_tokens"],
            timing["teacher_peak"],
            timing["student_peak"],
            accum_batches,
        )

    def _optimizer_step(
        self, epoch: int, window: dict, accum_batches: int, timing: dict
    ) -> None:
        args, dist_state, params = self.args, self.dist_state, self.params
        opt_start = self._stamp()
        # Partial final window: rescale grads so the average matches a full window.
        if accum_batches < args.grad_accum_steps:
            scale = args.grad_accum_steps / accum_batches
            for param in params:
                if param.grad is not None:
                    param.grad.mul_(scale)
        sync_gradients(params, dist_state)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()
        timing["optimizer"] += self._stamp() - opt_start
        self.step += 1

        metrics = torch.tensor(
            [
                window["loss"],
                window["teacher"],
                window["lm"],
                window["kd"],
                float(accum_batches),
            ],
            device=self.device,
            dtype=torch.float32,
        )
        if dist_state.dp_enabled:
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM, group=dist_state.dp_group)
        total_batches = float(metrics[4].item())
        loss = float(metrics[0].item()) / total_batches
        teacher = float(metrics[1].item()) / total_batches
        lm = float(metrics[2].item()) / total_batches
        kd = float(metrics[3].item()) / total_batches
        lr = self.optimizer.param_groups[0]["lr"]

        if dist_state.is_rank_zero:
            logger.info(
                "epoch %d step %d loss=%.4f teacher=%.4f lm=%.4f kd=%.4f grad_norm=%.3f",
                epoch,
                self.step,
                loss,
                teacher,
                lm,
                kd,
                float(grad_norm),
            )
            if self.wandb_run is not None:
                self.wandb_run.log(
                    {
                        "loss": loss,
                        "teacher_loss": teacher,
                        "lm_loss": lm,
                        "kd_loss": kd,
                        "grad_norm": float(grad_norm),
                        "lr": lr,
                        "epoch": epoch,
                    },
                    step=self.step,
                )
        if self.collect_timing and self.step % args.timing_log_every == 0:
            self._log_timing(timing, accum_batches)
        if args.save_every_steps and self.step % args.save_every_steps == 0:
            self._save_checkpoint(self.out_dir / f"checkpoint-{self.step}")

    def run(self, loader, sampler) -> None:
        args = self.args
        self.optimizer.zero_grad()
        for epoch in range(args.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            accum_batches = 0
            window = _new_window()
            timing = _new_timing()
            num_batches = len(loader)
            loader_iter = iter(loader)
            for batch_idx in range(num_batches):
                start = self._stamp()
                batch = next(loader_iter)
                timing["data"] += self._stamp() - start

                start = self._stamp()
                batch = {
                    k: v.to(self.device, non_blocking=True) for k, v in batch.items()
                }
                timing["h2d"] += self._stamp() - start

                will_step = (
                    accum_batches + 1 == args.grad_accum_steps
                    or batch_idx + 1 == num_batches
                )

                ga = args.grad_accum_steps
                cuda = self.device.type == "cuda"
                # Teacher phase: forward + backward. Freeing its graph here, before
                # the student forward, keeps peak VRAM at max(teacher, student) not sum.
                if cuda:
                    torch.cuda.reset_peak_memory_stats(self.device)
                t0 = self._stamp()
                teacher_loss, teacher_kd, positions = self.model.teacher_step(
                    batch["input_ids"],
                    labels=batch["labels"],
                    attention_mask=batch["attention_mask"],
                )
                t1 = self._stamp()
                if teacher_loss.requires_grad:
                    (args.teacher_loss_weight * teacher_loss / ga).backward()
                # Student phase: forward + backward (teacher graph already freed).
                # Reset peak between phases to attribute the bottleneck to one phase.
                t2 = self._stamp()
                if cuda:
                    timing["teacher_peak"] = max(
                        timing["teacher_peak"],
                        torch.cuda.max_memory_allocated(self.device) / 1024**3,
                    )
                    torch.cuda.reset_peak_memory_stats(self.device)
                lm_loss, kd_loss, s_calls, s_tokens = self.model.student_step(
                    batch["input_ids"],
                    labels=batch["labels"],
                    attention_mask=batch["attention_mask"],
                    teacher_kd=teacher_kd,
                    positions=positions,
                )
                t3 = self._stamp()
                ((args.lm_loss_weight * lm_loss + args.kd_loss_weight * kd_loss) / ga).backward()
                t4 = self._stamp()
                if cuda:
                    timing["student_peak"] = max(
                        timing["student_peak"],
                        torch.cuda.max_memory_allocated(self.device) / 1024**3,
                    )
                timing["forward"] += (t1 - t0) + (t3 - t2)
                timing["backward"] += (t2 - t1) + (t4 - t3)
                if self.collect_timing:
                    timing["student_calls"] += s_calls
                    timing["student_tokens"] += s_tokens

                accum_batches += 1
                t, l, k = teacher_loss.item(), lm_loss.item(), kd_loss.item()
                window["loss"] += (
                    args.teacher_loss_weight * t
                    + args.lm_loss_weight * l
                    + args.kd_loss_weight * k
                )
                window["teacher"] += t
                window["lm"] += l
                window["kd"] += k

                if will_step:
                    self._optimizer_step(epoch, window, accum_batches, timing)
                    accum_batches = 0
                    window = _new_window()
                    timing = _new_timing()

            self._save_checkpoint(self.out_dir / f"epoch-{epoch}")  # per-epoch history

        self._save_checkpoint(self.out_dir)  # final adapter (save_every_steps=0)


def main(args: TrainArgs) -> None:
    from transformers import AutoTokenizer

    if args.qlora and args.tp_size > 1:
        raise ValueError("qlora (bitsandbytes 4-bit) is incompatible with tp_size > 1")
    if args.grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be positive")

    dist_state = init_distributed(tp_size=args.tp_size, dp_size=args.dp_size)
    out_dir = Path(args.output_dir)

    # Bound before the try so the except can reference them on setup-time
    # crashes (None => skip the rescue save / wandb finish).
    model = None
    wandb_run = None
    try:
        torch.manual_seed(args.seed + dist_state.dp_rank)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        config = default_codi_config_from_tokenizer(
            tokenizer,
            latent_steps=args.latent_steps,
            kd_layers=tuple(args.kd_layers) if args.kd_layers else None,
            teacher_loss_weight=args.teacher_loss_weight,
            lm_loss_weight=args.lm_loss_weight,
            kd_loss_weight=args.kd_loss_weight,
        )
        model = _build_model(args, dist_state, config)

        pad_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
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
            build_workers=args.build_workers,
            data_source=args.data_source,
        )

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
        optimizer = bitsandbytes.optim.AdamW8bit(params, lr=args.lr, weight_decay=0.01)
        wandb_run = _setup_wandb(args, dist_state)

        trainer = _Trainer(
            args, dist_state, model, params, optimizer, out_dir, wandb_run
        )
        trainer.run(loader, sampler)

        if torch.cuda.is_available():
            peak_alloc = torch.cuda.max_memory_allocated(dist_state.device) / 1024**3
            peak_reserved = torch.cuda.max_memory_reserved(dist_state.device) / 1024**3
            logger.info(
                "rank %d/%d peak VRAM: %.1f GB allocated / %.1f GB reserved",
                dist_state.rank,
                dist_state.world_size,
                peak_alloc,
                peak_reserved,
            )

    except BaseException:
        logger.exception(
            "rank %d/%d failed; exiting without graceful distributed cleanup",
            dist_state.rank,
            dist_state.world_size,
        )
        if model is not None:
            try:
                _save_checkpoint(model, dist_state, out_dir / "crash")
            except Exception:
                logger.exception("crash-time checkpoint save failed")
        if wandb_run is not None:
            try:
                wandb_run.finish(exit_code=1)
            except Exception:
                logger.exception("wandb.finish() failed during crash handling")
        sys.stdout.flush()
        sys.stderr.flush()
        if dist_state.enabled:
            os._exit(1)
        raise
    else:
        destroy_distributed(dist_state)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from cwm.common.params import load_from_cli

    main(load_from_cli(TrainArgs))
