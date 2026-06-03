# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import nn

from cwm.training.codi_config import CodiConfig, apply_lora
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

        # The streaming pass is CPU/dispatch-bound: hundreds of sequential
        # decoder calls, each layer multiplied by the PEFT per-linear wrappers
        # and nn.Module machinery (~70% of the py-spy profile). Compile each
        # decoder layer in place rather than the whole stack -- that dispatch
        # hotspot lives inside one layer, so a single small per-layer graph
        # (reused across every layer and call) collapses the same Python overhead
        # while compiling in seconds and tolerating the streaming pass's varying
        # query width / KV-cache length far better than one giant dynamic graph
        # over all layers. lm_head and the outer model stay eager; `_call_model`
        # calls `causal_lm.model` directly and transparently runs the compiled
        # layers. dynamic=True because both the query width and the KV-cache
        # length change every call. Always on (the optimal path); set
        # TORCHDYNAMO_DISABLE=1 to fall back to eager (e.g. tests).
        causal_lm = self._causal_lm_module()
        if causal_lm is not None:
            layers = causal_lm.model.layers
            for i in range(len(layers)):
                layers[i] = torch.compile(layers[i], dynamic=True)

    @property
    def input_embeddings(self) -> nn.Module:
        return self.student.get_input_embeddings()

    def _causal_lm_module(self) -> nn.Module | None:
        base_model = getattr(self.student, "base_model", None)
        causal_lm = getattr(base_model, "model", None)
        if causal_lm is None:
            causal_lm = self.student
        if hasattr(causal_lm, "model") and hasattr(causal_lm, "lm_head"):
            return causal_lm
        return None

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

        # Per row, truncated to its real length so no padding enters the forward
        # (caps the hidden-state peak at one row instead of the full batch).
        teacher_kd: list[list[torch.Tensor]] | None = None
        for b in range(input_ids.shape[0]):
            row_mask = batch_idx == b
            if not row_mask.any():
                continue
            valid_len = int(attention_mask[b].sum().item())
            row_ids = input_ids[b : b + 1, :valid_len]
            causal_lm = self._causal_lm_module()
            if causal_lm is not None:
                out = causal_lm.model(
                    input_ids=row_ids,
                    output_hidden_states=True,
                    use_cache=False,
                )
                hidden_states = out.hidden_states
            else:
                out = self.student(
                    input_ids=row_ids,
                    output_hidden_states=True,
                    use_cache=False,
                )
                hidden_states = out.hidden_states
            if teacher_kd is None:
                teacher_kd = [[] for _ in hidden_states[1:]]
            positions = teacher_pos[row_mask]
            for layer_kd, h in zip(teacher_kd, hidden_states[1:], strict=True):
                layer_kd.extend(h[0, positions])
        return [torch.stack(layer).detach() for layer in teacher_kd]

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

        batch_idx, teacher_pos = self._teacher_positions(
            input_ids,
            labels,
            attention_mask,
        )
        (
            lm_loss,
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

        kd_loss = self._kd_loss(student_kd_vecs, teacher_kd_vecs, lm_loss)
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
