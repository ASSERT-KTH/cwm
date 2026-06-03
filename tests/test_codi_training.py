import torch
from transformers import CwmConfig, CwmForCausalLM

from cwm.training.codi import CodiModel
from cwm.training.codi_config import CodiConfig
from cwm.training.codi_streaming import streaming_student_outputs


def _tiny_model() -> CwmForCausalLM:
    config = CwmConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=2,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
        },
        layer_types=["full_attention", "full_attention"],
    )
    return CwmForCausalLM(config)


def test_codi_loss_backprops_only_to_student() -> None:
    torch.manual_seed(0)
    student = _tiny_model()

    config = CodiConfig(
        latent_span_start_token_id=10,
        latent_span_end_token_id=13,
        latent_steps=2,
        latent_start_token_id=11,
        latent_end_token_id=12,
    )
    model = CodiModel(student=student, config=config)

    input_ids = torch.tensor(
        [
            [2, 5, 10, 7, 13, 9, 3, 0],
            [2, 10, 13, 9, 4, 0, 0, 0],
        ]
    )
    attention_mask = input_ids.ne(0).long()
    labels = input_ids.masked_fill(~attention_mask.bool(), config.ignore_index)

    output = model(input_ids, labels=labels, attention_mask=attention_mask)

    assert output.kd_positions.shape == (2, 3)
    assert output.metrics["num_kd_positions"].item() == 2
    assert output.loss.requires_grad

    output.loss.backward()

    trainable = [(name, p) for name, p in model.student.named_parameters() if p.requires_grad]
    frozen = [(name, p) for name, p in model.student.named_parameters() if not p.requires_grad]
    assert trainable
    assert all("lora_" in name for name, _ in trainable)
    assert all(p.grad is not None for _, p in trainable)
    assert all(p.grad is None for _, p in frozen)


def _codi_model(latent_steps: int = 2) -> tuple[CodiModel, CodiConfig]:
    config = CodiConfig(
        latent_span_start_token_id=10,
        latent_span_end_token_id=13,
        latent_steps=latent_steps,
        latent_start_token_id=11,
        latent_end_token_id=12,
    )
    return CodiModel(student=_tiny_model(), config=config), config


def test_latent_span_replaces_inner_text() -> None:
    torch.manual_seed(0)
    model, config = _codi_model(latent_steps=2)
    model.eval()

    input_ids = torch.tensor([[2, 10, 7, 8, 13, 9]])
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    batch_idx, teacher_pos = model._teacher_positions(input_ids, labels, attention_mask)

    _, _, student_pos, _, student_tokens = streaming_student_outputs(
        model, input_ids, labels, attention_mask, batch_idx, teacher_pos
    )

    assert teacher_pos.tolist() == [4]
    assert student_pos.tolist() == [6]
    assert student_tokens == 8


def test_batched_streaming_matches_per_row() -> None:
    torch.manual_seed(0)
    model, config = _codi_model()
    model.eval()

    # Different lengths / span counts, each row carrying one KD target (token 13).
    rows = [
        [2, 5, 10, 7, 13, 9, 3, 0, 0, 0],
        [2, 10, 6, 6, 13, 9, 4, 8, 10, 3],
        [2, 13, 9, 10, 5, 0, 0, 0, 0, 0],
    ]
    input_ids = torch.tensor(rows)
    attention_mask = input_ids.ne(0).long()
    labels = input_ids.masked_fill(~attention_mask.bool(), config.ignore_index)

    batch_idx, teacher_pos = model._teacher_positions(input_ids, labels, attention_mask)
    _, kd_batched, spos_batched, _, _ = streaming_student_outputs(
        model, input_ids, labels, attention_mask, batch_idx, teacher_pos
    )

    # Reference: every row processed alone, concatenated row-major.
    ref_layers: list[list[torch.Tensor]] | None = None
    ref_spos: list[int] = []
    for b in range(len(rows)):
        ids, am = input_ids[b : b + 1], attention_mask[b : b + 1]
        lab = labels[b : b + 1]
        bi, tp = model._teacher_positions(ids, lab, am)
        _, kd_row, spos_row, _, _ = streaming_student_outputs(
            model, ids, lab, am, bi, tp
        )
        if ref_layers is None:
            ref_layers = [[] for _ in kd_row]
        for layer, vec in enumerate(kd_row):
            ref_layers[layer].append(vec)
        ref_spos.extend(spos_row.tolist())

    assert spos_batched.tolist() == ref_spos
    assert len(kd_batched) == len(ref_layers)
    for batched_layer, ref_parts in zip(kd_batched, ref_layers, strict=True):
        torch.testing.assert_close(batched_layer, torch.cat(ref_parts), atol=1e-4, rtol=1e-4)


def test_batched_lm_loss_invariant_to_duplication() -> None:
    torch.manual_seed(0)
    model, config = _codi_model()
    model.eval()

    row = [2, 5, 10, 7, 13, 9, 3]
    single = torch.tensor([row])
    dup = torch.tensor([row, row, row])

    losses = []
    for ids in (single, dup):
        am = torch.ones_like(ids)
        lab = ids.clone()
        losses.append(model(ids, labels=lab, attention_mask=am).lm_loss)

    torch.testing.assert_close(losses[0], losses[1], atol=1e-5, rtol=1e-5)


def test_checkpoint_single_call_matches_eager() -> None:
    # No <|line_sep|> -> a single student call, no latent block. The grad-enabled
    # (always-on) checkpointed path must give the same loss as the eager no_grad
    # plain-cache path, and grads must reach the LoRA params.
    torch.manual_seed(0)
    model, config = _codi_model()
    model.eval()

    input_ids = torch.tensor([[2, 5, 13, 9, 3]])
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    with torch.no_grad():  # grad disabled -> eager plain-cache path (no checkpoint)
        ref_loss = model(input_ids, labels=labels, attention_mask=attention_mask).loss

    model.zero_grad(set_to_none=True)
    out = model(input_ids, labels=labels, attention_mask=attention_mask)
    out.loss.backward()

    torch.testing.assert_close(out.loss.detach(), ref_loss, atol=1e-4, rtol=1e-4)
    with_grad = [n for n, p in model.named_parameters() if p.grad is not None]
    assert with_grad and all("lora_" in n for n in with_grad)


def test_checkpoint_preserves_latent_gradient() -> None:
    # With <|line_sep|> + latents the latent gradient reaches the loss only via
    # later tokens attending to the injected latents through the KV cache. The
    # block-checkpointed path rebuilds the cache without detaching it, so the loss
    # matches the eager path and grads reach the thought projector.
    torch.manual_seed(0)
    config = CodiConfig(
        latent_span_start_token_id=10,
        latent_span_end_token_id=13,
        latent_steps=2,
        latent_start_token_id=11,
        latent_end_token_id=12,
    )
    model = CodiModel(student=_tiny_model(), config=config, use_thought_projector=True)
    model.eval()

    input_ids = torch.tensor([[2, 5, 10, 7, 13, 9, 3, 10, 8, 13, 9, 4]])
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    with torch.no_grad():  # grad disabled -> eager plain-cache path (no checkpoint)
        ref_loss = model(input_ids, labels=labels, attention_mask=attention_mask).loss

    model.zero_grad(set_to_none=True)
    out = model(input_ids, labels=labels, attention_mask=attention_mask)
    out.loss.backward()

    torch.testing.assert_close(out.loss.detach(), ref_loss, atol=1e-4, rtol=1e-4)
    with_grad = {n for n, p in model.named_parameters() if p.grad is not None}
    assert any("thought_projector" in n for n in with_grad)
    assert any("lora_" in n for n in with_grad)


def test_codi_kd_positions_ignore_masked_prompt_tokens() -> None:
    torch.manual_seed(0)
    student = _tiny_model()

    config = CodiConfig(
        latent_span_start_token_id=10,
        latent_span_end_token_id=13,
        latent_steps=1,
        latent_start_token_id=11,
        latent_end_token_id=12,
    )
    model = CodiModel(student=student, config=config)

    input_ids = torch.tensor([[2, 13, 9, 10, 7, 13, 9, 3]])
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:, :4] = config.ignore_index

    output = model(input_ids, labels=labels, attention_mask=attention_mask)

    assert output.kd_positions.tolist() == [[0, 5, 7]]
