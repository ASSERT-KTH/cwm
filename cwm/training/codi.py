# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
import time
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
        student: nn.Module,
        config: CodiConfig,
        use_thought_projector: bool = False,
    ) -> None:
        super().__init__()
        self.student = apply_lora(student, config)
        self.config = config

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
        batch_idx: torch.Tensor,
        teacher_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None, torch.Tensor, int, int]:
        cfg = self.config
        device = input_ids.device
        batch_size = input_ids.shape[0]
        lm_loss_sum = self.input_embeddings.weight.sum() * 0.0
        lm_loss_count = torch.zeros((), device=device, dtype=torch.long)
        student_pos: list[int] = []
        student_kd_vecs: list[list[torch.Tensor]] | None = None
        student_model_calls = 0
        student_tokens = 0
        targets_by_batch = [
            teacher_pos[batch_idx == b].tolist() for b in range(batch_size)
        ]

        # Loop-invariants built once instead of per code line.
        ignore = labels.new_tensor(cfg.ignore_index)
        sot_embed = self._token_embed(cfg.sot_token_id, device)
        eot_embed = self._token_embed(cfg.eot_token_id, device)
        proj_device = proj_dtype = None
        if self.thought_projector is not None:
            proj_param = next(self.thought_projector.parameters())
            proj_device, proj_dtype = proj_param.device, proj_param.dtype

        # CPU copies so per-line chunk splitting needs no GPU->CPU sync.
        valid_lens = attention_mask.sum(dim=1).tolist()
        rows_cpu = input_ids.tolist()

        def add_lm_loss(logits: torch.Tensor, next_labels: torch.Tensor) -> None:
            nonlocal lm_loss_sum, lm_loss_count
            if logits.shape[0] == 0:
                return
            lm_loss_sum = lm_loss_sum + F.cross_entropy(
                logits,
                next_labels,
                ignore_index=cfg.ignore_index,
                reduction="sum",
            )
            lm_loss_count += (next_labels != cfg.ignore_index).sum()

        def collect_student_kd(
            hidden_states: tuple[torch.Tensor, ...],
            offsets: list[int],
            start_pos: int,
        ) -> None:
            nonlocal student_kd_vecs
            if not offsets:
                return
            if student_kd_vecs is None:
                student_kd_vecs = [[] for _ in hidden_states[1:]]
            offset_tensor = torch.tensor(offsets, device=device)
            for layer_kd, hidden in zip(
                student_kd_vecs, hidden_states[1:], strict=True
            ):
                layer_kd.extend(hidden[0, offset_tensor])
            student_pos.extend(start_pos + offset for offset in offsets)

        for b in range(batch_size):
            valid_len = valid_lens[b]
            row_ids = rows_cpu[b][:valid_len]
            past = None
            prev_logits: torch.Tensor | None = None
            student_len = 0
            targets = targets_by_batch[b]
            target_idx = 0

            def step(
                embed: torch.Tensor,
                label: torch.Tensor,
                *,
                output_hidden_states: bool = True,
            ) -> torch.Tensor | None:
                nonlocal past, prev_logits, student_len, student_model_calls, student_tokens
                student_model_calls += 1
                student_tokens += 1
                output = self.student(
                    inputs_embeds=embed.view(1, 1, -1),
                    past_key_values=past,
                    output_hidden_states=output_hidden_states,
                    use_cache=True,
                )
                past = output.past_key_values
                if prev_logits is not None:
                    add_lm_loss(prev_logits.view(1, -1), label.view(1))
                prev_logits = output.logits[0, -1]
                student_len += 1
                if output_hidden_states:
                    return output.hidden_states[-1][0, -1]
                return None

            def step_chunk(
                embeds: torch.Tensor,
                chunk_labels: torch.Tensor,
                *,
                output_hidden_states: bool,
            ):
                nonlocal past, prev_logits, student_len, student_model_calls, student_tokens
                student_model_calls += 1
                student_tokens += embeds.shape[0]
                output = self.student(
                    inputs_embeds=embeds.unsqueeze(0),
                    past_key_values=past,
                    output_hidden_states=output_hidden_states,
                    use_cache=True,
                )
                past = output.past_key_values
                if prev_logits is not None:
                    add_lm_loss(prev_logits.view(1, -1), chunk_labels[:1])
                add_lm_loss(output.logits[0, :-1], chunk_labels[1:])
                prev_logits = output.logits[0, -1]
                student_len += embeds.shape[0]
                return output

            # Chunks end right after each line_sep (pure Python, no GPU sync).
            boundaries: list[tuple[int, int, bool]] = []
            prev = 0
            for i in range(valid_len):
                if row_ids[i] == cfg.line_sep_token_id:
                    boundaries.append((prev, i + 1, True))
                    prev = i + 1
            if prev < valid_len:
                boundaries.append((prev, valid_len, False))

            for orig_pos, end_pos, has_line_sep in boundaries:
                offsets = []
                while target_idx < len(targets) and targets[target_idx] < end_pos:
                    if targets[target_idx] >= orig_pos:
                        offsets.append(targets[target_idx] - orig_pos)
                    target_idx += 1

                start_pos = student_len
                chunk_ids = input_ids[b, orig_pos:end_pos]
                output = step_chunk(
                    self.input_embeddings(chunk_ids),
                    labels[b, orig_pos:end_pos],
                    output_hidden_states=bool(offsets),
                )
                if offsets:
                    collect_student_kd(output.hidden_states, offsets, start_pos)

                if not has_line_sep:
                    continue

                base = step(sot_embed, ignore)
                for _ in range(cfg.latent_steps):
                    latent = base
                    if self.thought_projector is not None:
                        latent = self.thought_projector(
                            latent.to(device=proj_device, dtype=proj_dtype)
                        )
                    base = step(latent, ignore)
                step(
                    eot_embed,
                    ignore,
                    output_hidden_states=False,
                )

        lm_loss = lm_loss_sum / lm_loss_count.clamp(min=1)
        stacked_kd = (
            [torch.stack(layer) for layer in student_kd_vecs]
            if student_kd_vecs is not None
            else None
        )
        return (
            lm_loss,
            stacked_kd,
            torch.tensor(student_pos, device=device, dtype=batch_idx.dtype),
            student_model_calls,
            student_tokens,
        )

    def _teacher_positions(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        return batch_idx[valid], teacher_pos[valid]

    def _hidden_loss(
        self,
        student_vec: torch.Tensor,
        teacher_vec: torch.Tensor,
    ) -> torch.Tensor:
        student_vec = student_vec.float()
        teacher_vec = teacher_vec.float()

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
        student_kd_vecs: list[torch.Tensor] | None,
        teacher_kd_vecs: list[torch.Tensor] | None,
        zero_ref: torch.Tensor,
    ) -> torch.Tensor:
        if student_kd_vecs is None:
            return zero_ref.sum() * 0.0

        losses = [
            self._hidden_loss(s_v, t_v)
            for s_v, t_v in zip(student_kd_vecs, teacher_kd_vecs, strict=True)
        ]
        return torch.stack(losses).mean()

    def _teacher_kd_vecs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        batch_idx: torch.Tensor,
        teacher_pos: torch.Tensor,
    ) -> list[torch.Tensor] | None:
        if batch_idx.numel() == 0:
            return None

        teacher_output = self.student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        return [
            h[batch_idx, teacher_pos].detach()
            for h in teacher_output.hidden_states[1:]
        ]

    def _timer(self, device: torch.device, enabled: bool) -> float:
        if enabled and device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        profile: bool = False,
    ) -> CodiOutput:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if labels is None:
            labels = input_ids.masked_fill(
                ~attention_mask.bool(),
                self.config.ignore_index,
            )

        batch_idx, teacher_pos = self._teacher_positions(
            input_ids,
            labels,
            attention_mask,
        )
        start = self._timer(input_ids.device, profile)
        (
            lm_loss,
            student_kd_vecs,
            student_pos,
            student_model_calls,
            student_tokens,
        ) = self._streaming_student_outputs(
            input_ids,
            labels,
            attention_mask,
            batch_idx,
            teacher_pos,
        )
        student_s = self._timer(input_ids.device, profile) - start

        with torch.no_grad():
            disable_adapter = getattr(self.student, "disable_adapter", None)
            if disable_adapter is None:
                raise RuntimeError(
                    "CodiModel needs a PEFT student with disable_adapter()."
                )
            was_training = self.student.training
            self.student.eval()
            try:
                with disable_adapter():
                    start = self._timer(input_ids.device, profile)
                    teacher_kd_vecs = self._teacher_kd_vecs(
                        input_ids,
                        attention_mask,
                        batch_idx,
                        teacher_pos,
                    )
                    teacher_s = self._timer(input_ids.device, profile) - start
            finally:
                self.student.train(was_training)

        start = self._timer(input_ids.device, profile)
        kd_loss = self._kd_loss(student_kd_vecs, teacher_kd_vecs, lm_loss)
        kd_s = self._timer(input_ids.device, profile) - start
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
            "time_student_s": torch.tensor(student_s, device=input_ids.device),
            "time_teacher_s": torch.tensor(teacher_s, device=input_ids.device),
            "time_kd_s": torch.tensor(kd_s, device=input_ids.device),
            "student_model_calls": torch.tensor(student_model_calls, device=input_ids.device),
            "student_tokens": torch.tensor(student_tokens, device=input_ids.device),
        }

        return CodiOutput(
            loss=loss,
            lm_loss=lm_loss,
            kd_loss=kd_loss,
            kd_positions=kd_positions,
            metrics=metrics,
        )
