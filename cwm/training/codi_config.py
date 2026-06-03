# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
from typing import Literal

from torch import nn


@dataclass
class CodiConfig:
    line_sep_token_id: int
    sot_token_id: int
    eot_token_id: int
    action_sep_token_id: int
    latent_steps: int
    lm_loss_weight: float = 1.0
    kd_loss_weight: float = 1.0
    ignore_index: int = -100
    distill_offset: int = 1
    expected_action_next_token_id: int | None = None
    kd_loss: Literal["l1", "smooth_l1", "mse"] = "l1"
    normalize_kd_by_teacher_std: bool = True
    kd_eps: float = 1e-6
    # Recompute each student call's activations in backward (Option A). Detaches
    # the KV cache at every call boundary; keeps the latent generation gradient.
    gradient_checkpointing: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    def __post_init__(self) -> None:
        if self.latent_steps < 0:
            raise ValueError("latent_steps must be non-negative")
        if self.distill_offset < 0:
            raise ValueError("distill_offset must be non-negative")
        if self.lora_r <= 0:
            raise ValueError("lora_r must be positive")
        if self.lora_alpha <= 0:
            raise ValueError("lora_alpha must be positive")
        if self.lora_dropout < 0:
            raise ValueError("lora_dropout must be non-negative")


def cwm_token_id(tokenizer, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is not None and token_id != tokenizer.unk_token_id:
        return int(token_id)

    encoded = tokenizer.encode(token, add_special_tokens=False)
    if len(encoded) != 1:
        raise ValueError(f"{token!r} is not a single token: {encoded}")
    return int(encoded[0])


def default_codi_config_from_tokenizer(
    tokenizer,
    *,
    latent_steps: int,
    require_action_next_eot: bool = True,
) -> CodiConfig:
    return CodiConfig(
        line_sep_token_id=cwm_token_id(tokenizer, "<|line_sep|>"),
        sot_token_id=cwm_token_id(tokenizer, "<|reasoning_thinking_start|>"),
        eot_token_id=cwm_token_id(tokenizer, "<|reasoning_thinking_end|>"),
        action_sep_token_id=cwm_token_id(tokenizer, "<|action_sep|>"),
        latent_steps=latent_steps,
        expected_action_next_token_id=cwm_token_id(tokenizer, "<|eot_id|>")
        if require_action_next_eot
        else None,
    )


def apply_lora(student: nn.Module, config: CodiConfig) -> nn.Module:
    if hasattr(student, "peft_config"):
        return student

    from peft import LoraConfig, TaskType, get_peft_model

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules=list(config.lora_target_modules),
    )
    return get_peft_model(student, lora_config)

