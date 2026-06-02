# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Literal

import torch
import torch.nn.functional as F
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


@dataclass
class CodiOutput:
    loss: torch.Tensor
    lm_loss: torch.Tensor
    kd_loss: torch.Tensor
    logits: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    orig_to_student_pos: torch.Tensor
    kd_positions: torch.Tensor
    metrics: dict[str, torch.Tensor]


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


def load_codi_teacher_student(
    model_name_or_path: str = "facebook/cwm",
    *,
    latent_steps: int,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str | dict | None = "auto",
    token: str | bool | None = True,
    use_thought_projector: bool = False,
) -> tuple[object, "CodiModel"]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token=token)
    config = default_codi_config_from_tokenizer(tokenizer, latent_steps=latent_steps)
    teacher = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        dtype=dtype,
        device_map=device_map,
        token=token,
    )
    student = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        dtype=dtype,
        device_map=device_map,
        token=token,
    )
    return tokenizer, CodiModel(
        teacher=teacher,
        student=student,
        config=config,
        use_thought_projector=use_thought_projector,
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


class CodiModel(nn.Module):
    def __init__(
        self,
        *,
        teacher: nn.Module,
        student: nn.Module,
        config: CodiConfig,
        use_thought_projector: bool = False,
    ) -> None:
        super().__init__()
        self.teacher = teacher
        self.student = apply_lora(student, config)
        self.config = config
        self.teacher.requires_grad_(False)
        self.teacher.eval()

        self.thought_projector = None
        if use_thought_projector:
            hidden_size = self.student.config.hidden_size
            self.thought_projector = nn.Sequential(
                nn.Linear(hidden_size, hidden_size, bias=False),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size, bias=False),
                nn.LayerNorm(hidden_size),
            )
            # Match student embedding device/dtype (sharded bf16 loading).
            ref = self.input_embeddings.weight
            self.thought_projector.to(device=ref.device, dtype=ref.dtype)

    @property
    def input_embeddings(self) -> nn.Module:
        return self.student.get_input_embeddings()

    def _token_embed(self, token_id: int, device: torch.device) -> torch.Tensor:
        token = torch.tensor([token_id], device=device)
        return self.input_embeddings(token)[0]

    def _streaming_student_outputs(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[object, torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        return:
            - `student_output`: output of the student model with TEACHER FORCING and LATENT REASONING.
                - TEACHER FORCING: If at i-th position, the student's output is different from ground truth, we will feed the ground truth token embedding at (i+1)-th step instead of the student's output.
                - LATENT REASONING: after each <line_sep_token_id>, we will fix the following tokens as "<reasoning_thinking_start> <latent_embedding for a fixed number> <reasoning_thinking_end>". The original tokens will be replaced.
            - `student_labels`: the labels for the student model, which is the same as `labels` except that the positions for latent reasoning are set to `ignore_index`.
            - `student_attention_mask`: the attention mask for the student model, which is 1 for valid tokens and 0 for padding tokens.
            - `orig_to_student_pos`: since the replacement of latent reasoning will change the sequence length, `orig_to_student_pos` is used to map the original input token indexes to the `student_labels` token indexes. 
        '''
        cfg = self.config
        device = input_ids.device
        batch_size = input_ids.shape[0]
        orig_to_student = torch.full_like(input_ids, -1)

        per_logits: list[torch.Tensor] = []
        per_hidden: list[list[torch.Tensor]] = []
        per_labels: list[torch.Tensor] = []

        for b in range(batch_size):
            valid_len = int(attention_mask[b].sum().item())
            past = None
            logits_steps: list[torch.Tensor] = []
            hidden_steps: list[list[torch.Tensor]] | None = None
            out_labels: list[torch.Tensor] = []

            def step(embed: torch.Tensor) -> torch.Tensor:
                nonlocal past, hidden_steps
                output = self.student(
                    inputs_embeds=embed.view(1, 1, -1),
                    past_key_values=past,
                    output_hidden_states=True,
                    use_cache=True,
                )
                past = output.past_key_values
                logits_steps.append(output.logits[0, -1])
                if hidden_steps is None:
                    hidden_steps = [[] for _ in output.hidden_states]
                for layer_idx, hidden in enumerate(output.hidden_states):
                    hidden_steps[layer_idx].append(hidden[0, -1])
                return output.hidden_states[-1][0, -1]

            def step_chunk(embeds: torch.Tensor) -> torch.Tensor:
                nonlocal past, hidden_steps
                output = self.student(
                    inputs_embeds=embeds.unsqueeze(0),
                    past_key_values=past,
                    output_hidden_states=True,
                    use_cache=True,
                )
                past = output.past_key_values
                logits_steps.extend(output.logits[0])
                if hidden_steps is None:
                    hidden_steps = [[] for _ in output.hidden_states]
                for layer_idx, hidden in enumerate(output.hidden_states):
                    hidden_steps[layer_idx].extend(hidden[0])
                return output.hidden_states[-1][0, -1]

            orig_pos = 0
            while orig_pos < valid_len:
                remaining = input_ids[b, orig_pos:valid_len]
                sep_positions = (remaining == cfg.line_sep_token_id).nonzero()
                if sep_positions.numel() == 0:
                    chunk_len = valid_len - orig_pos
                    has_line_sep = False
                else:
                    chunk_len = int(sep_positions[0].item()) + 1
                    has_line_sep = True

                start_pos = len(logits_steps)
                end_pos = orig_pos + chunk_len
                orig_to_student[b, orig_pos:end_pos] = torch.arange(
                    start_pos,
                    start_pos + chunk_len,
                    device=device,
                    dtype=orig_to_student.dtype,
                )

                # Optimization: run contiguous teacher-forced tokens as one
                # cached forward instead of launching one model call per token.
                chunk_ids = input_ids[b, orig_pos:end_pos]
                step_chunk(self.input_embeddings(chunk_ids))
                out_labels.extend(labels[b, orig_pos:end_pos])
                orig_pos = end_pos

                if not has_line_sep:
                    continue

                base = step(self._token_embed(cfg.sot_token_id, device))
                out_labels.append(labels.new_tensor(cfg.ignore_index))
                for _ in range(cfg.latent_steps):
                    latent = base
                    if self.thought_projector is not None:
                        p = next(self.thought_projector.parameters())
                        latent = self.thought_projector(latent.to(device=p.device, dtype=p.dtype))
                    base = step(latent)
                    out_labels.append(labels.new_tensor(cfg.ignore_index))
                step(self._token_embed(cfg.eot_token_id, device))
                out_labels.append(labels.new_tensor(cfg.ignore_index))

            per_logits.append(torch.stack(logits_steps))
            assert hidden_steps is not None
            per_hidden.append([torch.stack(layer) for layer in hidden_steps])
            per_labels.append(torch.stack(out_labels).long())

        max_len = max(x.shape[0] for x in per_logits)
        vocab_size = per_logits[0].shape[-1]
        hidden_size = per_hidden[0][0].shape[-1]
        num_layers = len(per_hidden[0])

        logits = per_logits[0].new_zeros(batch_size, max_len, vocab_size)
        student_labels = labels.new_full((batch_size, max_len), cfg.ignore_index)
        student_mask = attention_mask.new_zeros(batch_size, max_len)
        hidden_states = [
            per_hidden[0][0].new_zeros(batch_size, max_len, hidden_size)
            for _ in range(num_layers)
        ]

        for b in range(batch_size):
            seq_len = per_logits[b].shape[0]
            logits[b, :seq_len] = per_logits[b]
            student_labels[b, :seq_len] = per_labels[b]
            student_mask[b, :seq_len] = 1
            for layer_idx in range(num_layers):
                hidden_states[layer_idx][b, :seq_len] = per_hidden[b][layer_idx]

        student_output = SimpleNamespace(
            logits=logits,
            hidden_states=tuple(hidden_states),
        )
        return student_output, student_labels, student_mask, orig_to_student

    def _distill_positions(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        orig_to_student_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        batch_idx, action_pos = (input_ids == cfg.action_sep_token_id).nonzero(
            as_tuple=True
        )
        teacher_pos = action_pos + cfg.distill_offset
        valid = teacher_pos < input_ids.shape[1]
        batch_idx = batch_idx[valid]
        teacher_pos = teacher_pos[valid]

        valid = attention_mask[batch_idx, teacher_pos].bool()
        valid &= labels[batch_idx, teacher_pos] != cfg.ignore_index
        if cfg.expected_action_next_token_id is not None:
            valid &= input_ids[batch_idx, teacher_pos] == cfg.expected_action_next_token_id
        batch_idx = batch_idx[valid]
        teacher_pos = teacher_pos[valid]

        student_pos = orig_to_student_pos[batch_idx, teacher_pos]
        valid = student_pos >= 0
        return batch_idx[valid], teacher_pos[valid], student_pos[valid]

    def _lm_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] < 2:
            return logits.sum() * 0.0
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            labels[:, 1:].reshape(-1),
            ignore_index=self.config.ignore_index,
        )

    def _hidden_loss(
        self,
        student_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor,
        batch_idx: torch.Tensor,
        teacher_pos: torch.Tensor,
        student_pos: torch.Tensor,
    ) -> torch.Tensor:
        student_vec = student_hidden[batch_idx, student_pos].float()
        teacher_vec = teacher_hidden[batch_idx, teacher_pos].detach().float()

        if self.config.normalize_kd_by_teacher_std:
            scale = teacher_vec.std(dim=0, unbiased=False).clamp_min(self.config.kd_eps)
            student_vec = student_vec / scale
            teacher_vec = teacher_vec / scale

        if self.config.kd_loss == "l1":
            return F.l1_loss(student_vec, teacher_vec)
        if self.config.kd_loss == "smooth_l1":
            return F.smooth_l1_loss(student_vec, teacher_vec)
        if self.config.kd_loss == "mse":
            return F.mse_loss(student_vec, teacher_vec)
        raise ValueError(f"Unsupported kd_loss={self.config.kd_loss}")

    def _kd_loss(
        self,
        student_output,
        teacher_output,
        batch_idx: torch.Tensor,
        teacher_pos: torch.Tensor,
        student_pos: torch.Tensor,
    ) -> torch.Tensor:
        if batch_idx.numel() == 0:
            return student_output.logits.sum() * 0.0

        losses = [
            self._hidden_loss(s_h, t_h, batch_idx, teacher_pos, student_pos)
            for s_h, t_h in zip(
                student_output.hidden_states[1:],
                teacher_output.hidden_states[1:],
                strict=True,
            )
        ]
        return torch.stack(losses).mean()

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> CodiOutput:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if labels is None:
            labels = input_ids.masked_fill(
                ~attention_mask.bool(),
                self.config.ignore_index,
            )

        student_output, student_labels, student_mask, orig_to_student_pos = (
            self._streaming_student_outputs(input_ids, labels, attention_mask)
        )

        batch_idx, teacher_pos, student_pos = self._distill_positions(
            input_ids,
            labels,
            attention_mask,
            orig_to_student_pos,
        )
        lm_loss = self._lm_loss(student_output.logits, student_labels)

        with torch.no_grad():
            teacher_output = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        kd_loss = self._kd_loss(
            student_output,
            teacher_output,
            batch_idx,
            teacher_pos,
            student_pos,
        )
        loss = self.config.lm_loss_weight * lm_loss + self.config.kd_loss_weight * kd_loss
        kd_positions = torch.stack((batch_idx, teacher_pos, student_pos), dim=1)
        metrics = {
            "loss": loss.detach(),
            "lm_loss": lm_loss.detach(),
            "kd_loss": kd_loss.detach(),
            "num_kd_positions": torch.tensor(
                batch_idx.numel(),
                device=input_ids.device,
            ),
        }

        return CodiOutput(
            loss=loss,
            lm_loss=lm_loss,
            kd_loss=kd_loss,
            logits=student_output.logits,
            labels=student_labels,
            attention_mask=student_mask,
            orig_to_student_pos=orig_to_student_pos,
            kd_positions=kd_positions,
            metrics=metrics,
        )
