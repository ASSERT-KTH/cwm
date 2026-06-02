# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Teacher-forcing CODI distillation on CRUXEval-O execution traces.

Loads a HuggingFace CWM checkpoint as frozen teacher + LoRA student, builds the
ground-truth trace dataset in memory (``cwm.training.data``), and optimizes the
combined LM + hidden-state KD loss from ``CodiModel``. The student is
teacher-forced (ground-truth embeddings fed at every step).

    python -m cwm.training.train model_name_or_path=facebook/cwm output_dir=./codi-cruxeval
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

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
    grad_accum_steps: int = 8
    max_grad_norm: float = 1.0
    n_samples: int = -1
    max_seq_len: int = 8192
    seed: int = 42
    wandb_log: bool = False
    device_map: str = "auto"


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

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Trace format: the token after <|action_sep|> is source (not <|eot_id|>),
    # so KD positions must not require an eot next token.
    config = default_codi_config_from_tokenizer(
        tokenizer, latent_steps=args.latent_steps, require_action_next_eot=False
    )
    config.wandb_log = args.wandb_log

    load = lambda: AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, dtype=torch.bfloat16, device_map=args.device_map
    )
    model = CodiModel(
        teacher=load(),
        student=load(),
        config=config,
        use_thought_projector=args.use_thought_projector,
    )
    model.train()

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    device = model.input_embeddings.weight.device

    dataset = build_dataset(tokenizer, n_samples=args.n_samples, max_seq_len=args.max_seq_len)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: _collate(b, pad_id)
    )
    if args.grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be positive")

    params = [p for p in model.parameters() if p.requires_grad]
    logger.info("%d examples, %d trainable params", len(dataset), sum(p.numel() for p in params))
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    if args.wandb_log:
        import wandb

        wandb.init(project="codi-cruxeval", config=vars(args))

    step = 0

    def optimizer_step(epoch: int, out, accum_batches: int) -> None:
        nonlocal step
        if accum_batches < args.grad_accum_steps:
            scale = args.grad_accum_steps / accum_batches
            for param in params:
                if param.grad is not None:
                    param.grad.mul_(scale)
        torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        step += 1
        logger.info(
            "epoch %d step %d loss=%.4f lm=%.4f kd=%.4f kd_pos=%d",
            epoch, step, out.loss.item(), out.lm_loss.item(), out.kd_loss.item(),
            int(out.metrics["num_kd_positions"].item()),
        )

    optimizer.zero_grad()
    accum_batches = 0
    last_out = None
    for epoch in range(args.epochs):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["input_ids"], labels=batch["labels"],
                        attention_mask=batch["attention_mask"], wandb_step=step)
            (out.loss / args.grad_accum_steps).backward()
            accum_batches += 1
            last_out = out

            if accum_batches == args.grad_accum_steps:
                optimizer_step(epoch, out, accum_batches)
                accum_batches = 0
                last_out = None

        if accum_batches:
            assert last_out is not None
            optimizer_step(epoch, last_out, accum_batches)
            accum_batches = 0
            last_out = None

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.student.save_pretrained(out_dir)  # LoRA adapter only
    if model.thought_projector is not None:
        torch.save(model.thought_projector.state_dict(), out_dir / "thought_projector.pt")
    logger.info("Saved to %s", out_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from cwm.common.params import load_from_cli

    main(load_from_cli(TrainArgs))
