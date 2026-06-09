# Copyright (c) Meta Platforms, Inc. and affiliates.

from contextlib import contextmanager
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import nn

from cwm.training.codi_config import CodiConfig, KdVecs, apply_lora, select_kd_layers
from cwm.training.codi_streaming import streaming_student_outputs


@dataclass
class CodiOutput:
    loss: torch.Tensor
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

    @contextmanager
    def _teacher_mode(self):
        """Run the original model (without adapter) without gradients as the teacher."""
        disable_adapter = getattr(self.student, "disable_adapter", None)
        if disable_adapter is None:
            raise RuntimeError("CodiModel needs a PEFT student with disable_adapter().")
        was_training = self.student.training
        self.student.eval()
        try:
            with torch.no_grad(), disable_adapter():
                yield
        finally:
            self.student.train(was_training)

    def _teacher_kd_vecs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        batch_idx: torch.Tensor,
        teacher_pos: torch.Tensor,
    ) -> KdVecs | None:
        if batch_idx.numel() == 0:
            return None

        backbone = self.student.base_model.model.model
        num_kd_layers = (
            len(self.config.kd_layers)
            if self.config.kd_layers is not None
            else backbone.config.num_hidden_layers
        )
        collected_per_layer: list[list[torch.Tensor]] = [
            [] for _ in range(num_kd_layers)
        ]

        for row in range(input_ids.shape[0]):
            sample_kd_mask = batch_idx == row
            if not sample_kd_mask.any():
                continue

            # Right-padded, sum of attention_mask = number of real tokens.
            real_token_count = int(attention_mask[row].sum().item())
            backbone_output = backbone(
                input_ids=input_ids[row : row + 1, :real_token_count],
                output_hidden_states=True,
                use_cache=False,
            )

            kd_hidden_states = select_kd_layers(
                backbone_output.hidden_states, self.config.kd_layers
            )
            kd_token_positions = teacher_pos[sample_kd_mask]

            for layer_index, layer_hidden_state in enumerate(kd_hidden_states):
                collected_per_layer[layer_index].append(
                    layer_hidden_state[0, kd_token_positions]
                )

        vecs = [
            torch.cat(layer_chunks).detach() for layer_chunks in collected_per_layer
        ]
        return KdVecs(row_col=torch.stack([batch_idx, teacher_pos], dim=1), vecs=vecs)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> CodiOutput:
        batch_idx, teacher_pos = self._teacher_positions(
            input_ids, labels, attention_mask
        )

        # student forward, with latent reasoning injection.
        (
            lm_sum,
            lm_count,
            student_kd,
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

        # teacher forward, without adapter and gradient tracking.
        with self._teacher_mode():
            teacher_kd = self._teacher_kd_vecs(
                input_ids, attention_mask, batch_idx, teacher_pos
            )

        lm_loss = lm_sum / lm_count.clamp(min=1)
        kd_loss = self._kd_loss(student_kd, teacher_kd, lm_loss)

        loss = (
            self.config.lm_loss_weight * lm_loss + self.config.kd_loss_weight * kd_loss
        )

        metrics = {
            "loss": loss.detach(),
            "lm_loss": lm_loss.detach(),
            "kd_loss": kd_loss.detach(),
            "student_model_calls": torch.tensor(
                student_model_calls, device=input_ids.device
            ),
            "student_tokens": torch.tensor(student_tokens, device=input_ids.device),
        }

        return CodiOutput(
            loss=loss,
            lm_loss=lm_loss,
            kd_loss=kd_loss,
            metrics=metrics,
        )
