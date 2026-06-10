# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


@dataclass
class CodiConfig:
    latent_span_start_token_id: int
    latent_span_end_token_id: int
    latent_steps: int
    latent_start_token_id: int | None = None
    latent_end_token_id: int | None = None
    # CODI three-term loss: alpha*L_teacher + beta*L_student + gamma*L_KD.
    teacher_loss_weight: float = 1.0  # alpha: explicit-CoT CE on the full trace
    lm_loss_weight: float = 1.0       # beta: implicit-CoT (latent) student CE
    kd_loss_weight: float = 1.0       # gamma: hidden-state distillation
    ignore_index: int = -100
    kd_loss: Literal["l1", "smooth_l1", "mse"] = "l1"
    normalize_kd_by_teacher_std: bool = True
    kd_eps: float = 1e-6
    kd_layers: tuple[int, ...] | None = (-1,)
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
        if self.teacher_loss_weight < 0:
            raise ValueError("teacher_loss_weight must be non-negative")
        if self.lm_loss_weight < 0:
            raise ValueError("lm_loss_weight must be non-negative")
        if self.kd_loss_weight < 0:
            raise ValueError("kd_loss_weight must be non-negative")
        if self.lora_r <= 0:
            raise ValueError("lora_r must be positive")
        if self.lora_alpha <= 0:
            raise ValueError("lora_alpha must be positive")
        if self.lora_dropout < 0:
            raise ValueError("lora_dropout must be non-negative")


@dataclass
class KdVecs:
    row_col: torch.Tensor       # [N, 2] — (batch_row, teacher_col) for each KD position
    vecs: list[torch.Tensor]    # list[Tensor[N, H]], one per KD layer


def select_kd_layers(hidden_states, kd_layers: tuple[int, ...] | None):
    layers = hidden_states[1:]
    if kd_layers is None:
        return layers
    return tuple(layers[i] for i in kd_layers)


def kd_layers_are_last_only(
    kd_layers: tuple[int, ...] | None, num_hidden_layers: int
) -> bool:
    return kd_layers is not None and len(kd_layers) == 1 and kd_layers[0] in (
        -1,
        num_hidden_layers - 1,
    )


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
    kd_layers: tuple[int, ...] | None = (-1,),
    teacher_loss_weight: float = 1.0,
    lm_loss_weight: float = 1.0,
    kd_loss_weight: float = 1.0,
) -> CodiConfig:
    return CodiConfig(
        latent_span_start_token_id=cwm_token_id(tokenizer, "<|line_sep|>"),
        latent_span_end_token_id=cwm_token_id(tokenizer, "<|action_sep|>"),
        latent_start_token_id=cwm_token_id(tokenizer, "<|reasoning_thinking_start|>"),
        latent_end_token_id=cwm_token_id(tokenizer, "<|reasoning_thinking_end|>"),
        latent_steps=latent_steps,
        kd_layers=kd_layers,
        teacher_loss_weight=teacher_loss_weight,
        lm_loss_weight=lm_loss_weight,
        kd_loss_weight=kd_loss_weight,
    )


def apply_lora(student: nn.Module, config: CodiConfig) -> nn.Module:
    from peft import LoraConfig, TaskType, get_peft_model

    if getattr(student, "is_loaded_in_4bit", False):
        # QLoRA. HF checkpointing OFF: codi_streaming checkpoints itself and needs
        # use_cache=True, which gradient_checkpointing_enable() would force off.
        from peft import prepare_model_for_kbit_training

        student = prepare_model_for_kbit_training(
            student, use_gradient_checkpointing=False
        )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules=list(config.lora_target_modules),
    )
    return get_peft_model(student, lora_config)

