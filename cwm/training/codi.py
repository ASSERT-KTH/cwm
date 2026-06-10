# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import nn

from cwm.training.codi_config import (
    CodiConfig,
    KdVecs,
    apply_lora,
    kd_layers_are_last_only,
    select_kd_layers,
)
from cwm.training.codi_streaming import streaming_student_outputs


@dataclass
class CodiOutput:
    loss: torch.Tensor
    teacher_loss: torch.Tensor
    lm_loss: torch.Tensor
    kd_loss: torch.Tensor
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
        """
        Return positioins of latent_span_end_token_id tokens that are not masked out and not ignored by the loss.
        (batch_idx[i], teacher_pos[i]) means a latent_span_end_token_id token locates at teacher_pos[i] of sample batch_idx[i].
        """
        cfg = self.config
        batch_idx, teacher_pos = (input_ids == cfg.latent_span_end_token_id).nonzero(
            as_tuple=True
        )
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
        student_kd: KdVecs | None,
        teacher_kd: KdVecs | None,
        zero_ref: torch.Tensor,
    ) -> torch.Tensor:
        if student_kd is None:
            return zero_ref.sum() * 0.0

        assert torch.equal(
            student_kd.row_col, teacher_kd.row_col
        ), f"KD position mismatch: student {student_kd.row_col.tolist()} vs teacher {teacher_kd.row_col.tolist()}"
        losses = [
            self._hidden_loss(s, t)
            for s, t in zip(student_kd.vecs, teacher_kd.vecs, strict=True)
        ]
        return torch.stack(losses).mean()

    def _teacher_row_forward(self, ids: torch.Tensor, row_labels: torch.Tensor):
        """Full-sequence teacher forward for one row (gradient-checkpointed).

        Returns the row's summed teacher CE (with grad) plus the per-KD-layer
        hidden states detached (KD targets get stop-gradient, CODI's sg[.]).
        """
        causal_lm = self.student.base_model.model
        last_kd_layer_only = kd_layers_are_last_only(
            self.config.kd_layers, causal_lm.config.num_hidden_layers
        )
        out = causal_lm.model(
            input_ids=ids,
            output_hidden_states=not last_kd_layer_only,
            use_cache=False,
        )
        logits = causal_lm.lm_head(out.last_hidden_state)
        # Next-token CE over the full explicit trace (L_teacher).
        loss = F.cross_entropy(
            logits[0, :-1],
            row_labels[1:],
            ignore_index=self.config.ignore_index,
            reduction="sum",
        )
        kd_hidden_states = (
            (out.last_hidden_state,)
            if last_kd_layer_only
            else select_kd_layers(out.hidden_states, self.config.kd_layers)
        )
        return (loss, *(h.detach() for h in kd_hidden_states))

    def _teacher_forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        batch_idx: torch.Tensor,
        teacher_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, KdVecs | None]:
        """CODI teacher task: the SHARED model reads the explicit trace.

        Runs WITH gradients (teacher CE co-trains the same weights as the
        student); only the KD target hidden states are detached. Returns
        (teacher_loss_sum, teacher_token_count, teacher_kd).
        """
        num_kd_layers = (
            len(self.config.kd_layers)
            if self.config.kd_layers is not None
            else self.student.base_model.model.config.num_hidden_layers
        )
        collected_per_layer: list[list[torch.Tensor]] = [
            [] for _ in range(num_kd_layers)
        ]
        loss_sum = self.input_embeddings.weight.sum() * 0.0
        token_count = torch.zeros((), device=input_ids.device, dtype=torch.long)

        # Per-layer checkpointing: backward recomputes one decoder layer at a
        # time (peak ~1 layer, not 64). Off after the loop — it forces
        # use_cache=False, which would break the student's KV streaming.
        backbone = self.student.base_model.model.model
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        try:
            for row in range(input_ids.shape[0]):
                # Right-padded, sum of attention_mask = number of real tokens.
                real_token_count = int(attention_mask[row].sum().item())
                ids = input_ids[row : row + 1, :real_token_count]
                row_labels = labels[row, :real_token_count]
                loss, *kd_hidden_states = self._teacher_row_forward(ids, row_labels)
                loss_sum = loss_sum + loss
                token_count += (row_labels[1:] != self.config.ignore_index).sum()

                sample_kd_mask = batch_idx == row
                if not sample_kd_mask.any():
                    continue
                kd_token_positions = teacher_pos[sample_kd_mask]
                for layer_index, layer_hidden_state in enumerate(kd_hidden_states):
                    collected_per_layer[layer_index].append(
                        layer_hidden_state[0, kd_token_positions]
                    )
        finally:
            backbone.gradient_checkpointing_disable()

        teacher_kd = None
        if batch_idx.numel() > 0:
            vecs = [torch.cat(chunks) for chunks in collected_per_layer]
            teacher_kd = KdVecs(
                row_col=torch.stack([batch_idx, teacher_pos], dim=1), vecs=vecs
            )
        return loss_sum, token_count, teacher_kd

    def teacher_step(
        self,
        input_ids: torch.Tensor,
        *,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, KdVecs | None, tuple[torch.Tensor, torch.Tensor]]:
        """Teacher task (explicit trace, WITH grad); KD target detached. Split from
        the student so the caller can backward and free the teacher graph BEFORE the
        student forward, keeping peak VRAM = max(teacher, student) not their sum."""
        batch_idx, teacher_pos = self._teacher_positions(
            input_ids, labels, attention_mask
        )
        teacher_sum, teacher_count, teacher_kd = self._teacher_forward(
            input_ids, labels, attention_mask, batch_idx, teacher_pos
        )
        teacher_loss = teacher_sum / teacher_count.clamp(min=1)
        return teacher_loss, teacher_kd, (batch_idx, teacher_pos)

    def student_step(
        self,
        input_ids: torch.Tensor,
        *,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        teacher_kd: KdVecs | None,
        positions: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        """Student task (latent CoT streaming); KD aligned to the detached teacher."""
        batch_idx, teacher_pos = positions
        lm_sum, lm_count, student_kd, calls, tokens = streaming_student_outputs(
            self, input_ids, labels, attention_mask, batch_idx, teacher_pos
        )
        lm_loss = lm_sum / lm_count.clamp(min=1)
        kd_loss = self._kd_loss(student_kd, teacher_kd, lm_loss)
        return lm_loss, kd_loss, calls, tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> CodiOutput:
        teacher_loss, teacher_kd, positions = self.teacher_step(
            input_ids, labels=labels, attention_mask=attention_mask
        )
        lm_loss, kd_loss, calls, tokens = self.student_step(
            input_ids,
            labels=labels,
            attention_mask=attention_mask,
            teacher_kd=teacher_kd,
            positions=positions,
        )
        # CODI: L = alpha*L_teacher + beta*L_student + gamma*L_KD.
        loss = (
            self.config.teacher_loss_weight * teacher_loss
            + self.config.lm_loss_weight * lm_loss
            + self.config.kd_loss_weight * kd_loss
        )
        metrics = {
            "loss": loss.detach(),
            "teacher_loss": teacher_loss.detach(),
            "lm_loss": lm_loss.detach(),
            "kd_loss": kd_loss.detach(),
            "student_model_calls": torch.tensor(calls, device=input_ids.device),
            "student_tokens": torch.tensor(tokens, device=input_ids.device),
        }
        return CodiOutput(loss, teacher_loss, lm_loss, kd_loss, metrics)
