# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
from typing import Literal

from torch import nn


@dataclass
class CodiConfig:
    latent_span_start_token_id: int
    latent_span_end_token_id: int
    latent_steps: int
    latent_start_token_id: int | None = None
    latent_end_token_id: int | None = None
    lm_loss_weight: float = 1.0
    kd_loss_weight: float = 1.0
    ignore_index: int = -100
    distill_offset: int = 0
    kd_loss: Literal["l1", "smooth_l1", "mse"] = "l1"
    normalize_kd_by_teacher_std: bool = True
    kd_eps: float = 1e-6
    kd_layers: tuple[int, ...] | None = None
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


def select_kd_layers(hidden_states, kd_layers: tuple[int, ...] | None):
    """Per-layer hidden states to distill (drops the embedding layer at index 0).

    ``kd_layers`` selects a subset by index (negatives allowed, e.g. ``(-1,)`` for
    last layer only); ``None`` keeps every transformer layer. Teacher and student
    must pass the same ``kd_layers`` so the per-layer KD lists line up.
    """
    layers = hidden_states[1:]
    if kd_layers is None:
        return layers
    return tuple(layers[i] for i in kd_layers)


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
    kd_layers: tuple[int, ...] | None = None,
) -> CodiConfig:
    return CodiConfig(
        latent_span_start_token_id=cwm_token_id(tokenizer, "<|line_sep|>"),
        latent_span_end_token_id=cwm_token_id(tokenizer, "<|action_sep|>"),
        latent_start_token_id=cwm_token_id(tokenizer, "<|reasoning_thinking_start|>"),
        latent_end_token_id=cwm_token_id(tokenizer, "<|reasoning_thinking_end|>"),
        latent_steps=latent_steps,
        kd_layers=kd_layers,
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

