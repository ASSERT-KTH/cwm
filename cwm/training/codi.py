# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import nn

from cwm.training.codi_config import CodiConfig, apply_lora, select_kd_layers
from cwm.training.codi_streaming import streaming_student_outputs


@dataclass
class CodiOutput:
    loss: torch.Tensor
    lm_loss: torch.Tensor
    kd_loss: torch.Tensor
    kd_positions: torch.Tensor
    metrics: dict[str, torch.Tensor]


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

    def _teacher_positions(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        batch_idx, end_pos = (input_ids == cfg.latent_span_end_token_id).nonzero(
            as_tuple=True
        )
        teacher_pos = end_pos + cfg.distill_offset
        valid = teacher_pos < input_ids.shape[1]
        batch_idx = batch_idx[valid]
        teacher_pos = teacher_pos[valid]

        valid = attention_mask[batch_idx, teacher_pos].bool()
        valid &= labels[batch_idx, teacher_pos] != cfg.ignore_index
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

        # Per row, truncated to its real length so no padding enters the forward
        # (caps the hidden-state peak at one row instead of the full batch).
        # Store only detached KD slices; the large per-row hidden-state tuple can
        # then be released before the student graph is built.
        teacher_kd: list[list[torch.Tensor]] | None = None
        for b in range(input_ids.shape[0]):
            row_mask = batch_idx == b
            if not row_mask.any():
                continue
            valid_len = int(attention_mask[b].sum().item())
            row_ids = input_ids[b : b + 1, :valid_len]
            # Decoder-only forward (skips lm_head).
            causal_lm = self.student.base_model.model  # CwmForCausalLM
            out = causal_lm.model(input_ids=row_ids, output_hidden_states=True, use_cache=False)
            layers = select_kd_layers(out.hidden_states, self.config.kd_layers)
            if teacher_kd is None:
                teacher_kd = [[] for _ in layers]
            positions = teacher_pos[row_mask]
            for layer_kd, h in zip(teacher_kd, layers, strict=True):
                layer_kd.append(h[0, positions].detach())
            del out, layers
        return [torch.cat(layer, dim=0) for layer in teacher_kd]

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> CodiOutput:
        batch_idx, teacher_pos = self._teacher_positions(
            input_ids,
            labels,
            attention_mask,
        )
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
                    teacher_kd_vecs = self._teacher_kd_vecs(
                        input_ids,
                        attention_mask,
                        batch_idx,
                        teacher_pos,
                    )
            finally:
                self.student.train(was_training)

        (
            lm_sum,
            lm_count,
            student_kd_vecs,
            student_pos,
            student_model_calls,
            student_tokens,
        ) = streaming_student_outputs(
            self,
            input_ids,
            labels,
            attention_mask,
            batch_idx,
            teacher_pos,
        )

        lm_loss = lm_sum / lm_count.clamp(min=1)
        kd_loss = self._kd_loss(student_kd_vecs, teacher_kd_vecs, lm_loss)
        # Local per-token/position mean; grads are averaged across DP ranks in
        # sync_gradients. The sampler equalizes each rank's per-window token load,
        # so this plain rank-mean matches the global token-mean.
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
