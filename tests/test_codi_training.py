import torch
from transformers import CwmConfig, CwmForCausalLM

from cwm.training.codi import CodiModel
from cwm.training.codi_config import CodiConfig
from cwm.training.data import IGNORE_INDEX, MegabatchSortishSampler, build_dataset, build_example
from cwm.training.codi_streaming import streaming_student_outputs


class _FakeTok:
    """Minimal stand-in: build_example only needs bos_token_id + encode()."""

    bos_token_id = 1

    def encode(self, s, add_special_tokens=False):  # noqa: ANN001
        return [ord(c) % 50 + 2 for c in s]


def test_build_example_skips_impure_keeps_pure():
    tok = _FakeTok()
    pure = build_example("def f(x):\n    return x + 1\n", "5", tok, max_seq_len=8192)
    assert pure is not None
    ids, labels = pure
    assert len(ids) == len(labels)
    assert labels[0] == IGNORE_INDEX  # prompt tokens are masked
    assert build_example("def f(x):\n    print(x)\n    return x\n", "5", tok, max_seq_len=8192) is None


def test_build_dataset_drops_runaway_without_hanging():
    tok = _FakeTok()
    rows = [
        {"code": "def f(x):\n    return x * 2\n", "input": "3"},               # pure, fast
        {"code": "def f(x):\n    while True:\n        pass\n", "input": "1"},   # runaway
    ]
    out = build_dataset(rows, tok)
    assert len(out) == 1  # runaway dropped via SIGALRM, pure kept


def test_build_dataset_parallel_matches_serial():
    # Forked workers inherit the tokenizer COW; ordered imap -> identical result.
    tok = _FakeTok()
    rows = [{"code": f"def f(x):\n    return x + {k}\n", "input": "5"} for k in range(6)]
    assert build_dataset(rows, tok, workers=2) == build_dataset(rows, tok, workers=0)


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


def test_codi_loss_backprops_to_shared_lora() -> None:
    # CODI: teacher == student (shared base+LoRA). Both the teacher CE and the
    # student CE co-train the SAME LoRA params; the frozen base stays grad-free.
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

    batch_idx, _ = model._teacher_positions(input_ids, labels, attention_mask)
    assert batch_idx.numel() == 2  # one valid latent_span_end per row
    assert output.loss.requires_grad
    # Teacher CE is a live, finite term that co-trains the shared weights.
    assert output.teacher_loss.requires_grad
    assert torch.isfinite(output.teacher_loss) and output.teacher_loss > 0

    output.loss.backward()

    trainable = [(name, p) for name, p in model.student.named_parameters() if p.requires_grad]
    frozen = [(name, p) for name, p in model.student.named_parameters() if not p.requires_grad]
    assert trainable
    assert all("lora_" in name for name, _ in trainable)
    assert all(p.grad is not None for _, p in trainable)
    assert all(p.grad is None for _, p in frozen)


def test_teacher_shares_weights_and_kd_target_detached() -> None:
    # Teacher forward runs WITH grad (co-evolution); only the KD target hidden
    # states are stop-gradient'd (CODI's sg[.]).
    torch.manual_seed(0)
    model, _ = _codi_model()
    model.eval()  # deterministic (no LoRA dropout) for a clean check

    input_ids = torch.tensor([[2, 5, 10, 7, 13, 9, 3]])
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    batch_idx, teacher_pos = model._teacher_positions(input_ids, labels, attention_mask)
    t_sum, t_count, t_kd = model._teacher_forward(
        input_ids, labels, attention_mask, batch_idx, teacher_pos
    )

    assert t_sum.requires_grad  # teacher CE backprops into the shared LoRA
    assert int(t_count) > 0
    assert t_kd is not None
    assert all(not v.requires_grad for v in t_kd.vecs)  # KD target detached


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

    _, _, student_kd, _, student_tokens = streaming_student_outputs(
        model, input_ids, labels, attention_mask, batch_idx, teacher_pos
    )

    assert teacher_pos.tolist() == [4]
    assert student_kd.row_col[:, 1].tolist() == [4]  # KD registered at the teacher col
    assert student_tokens == 8  # 6 input tokens + injected latents (start/2 steps/end)


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
    _, _, kd_batched, _, _ = streaming_student_outputs(
        model, input_ids, labels, attention_mask, batch_idx, teacher_pos
    )

    # Reference: every row processed alone, concatenated row-major.
    ref_layers: list[list[torch.Tensor]] | None = None
    ref_spos: list[int] = []
    for b in range(len(rows)):
        ids, am = input_ids[b : b + 1], attention_mask[b : b + 1]
        lab = labels[b : b + 1]
        bi, tp = model._teacher_positions(ids, lab, am)
        _, _, kd_row, _, _ = streaming_student_outputs(
            model, ids, lab, am, bi, tp
        )
        if ref_layers is None:
            ref_layers = [[] for _ in kd_row.vecs]
        for layer, vec in enumerate(kd_row.vecs):
            ref_layers[layer].append(vec)
        ref_spos.extend(kd_row.row_col[:, 1].tolist())

    assert kd_batched.row_col[:, 1].tolist() == ref_spos
    assert len(kd_batched.vecs) == len(ref_layers)
    for batched_layer, ref_parts in zip(kd_batched.vecs, ref_layers, strict=True):
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
    # No <|line_sep|> -> a single student call, no latent block. The forward loss
    # under no_grad (checkpoint runs without recompute) must match the grad-enabled
    # forward+backward loss, and grads must reach the LoRA params.
    torch.manual_seed(0)
    model, config = _codi_model()
    model.eval()

    input_ids = torch.tensor([[2, 5, 13, 9, 3]])
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    with torch.no_grad():  # grad disabled -> checkpoint runs forward without recompute
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
    # block checkpoint rebuilds the cache without detaching it, so the no_grad
    # forward loss matches and grads reach the thought projector.
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

    with torch.no_grad():  # grad disabled -> checkpoint runs forward without recompute
        ref_loss = model(input_ids, labels=labels, attention_mask=attention_mask).loss

    model.zero_grad(set_to_none=True)
    out = model(input_ids, labels=labels, attention_mask=attention_mask)
    out.loss.backward()

    torch.testing.assert_close(out.loss.detach(), ref_loss, atol=1e-4, rtol=1e-4)
    with_grad = {n for n, p in model.named_parameters() if p.grad is not None}
    assert any("thought_projector" in n for n in with_grad)
    assert any("lora_" in n for n in with_grad)


def test_kd_layers_subset_restricts_distilled_layers() -> None:
    torch.manual_seed(0)
    config = CodiConfig(
        latent_span_start_token_id=10,
        latent_span_end_token_id=13,
        latent_steps=2,
        latent_start_token_id=11,
        latent_end_token_id=12,
        kd_layers=(-1,),  # last transformer layer only
    )
    model = CodiModel(student=_tiny_model(), config=config)

    input_ids = torch.tensor([[2, 5, 10, 7, 13, 9, 3]])
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    _, _, kd, _, _ = streaming_student_outputs(
        model, input_ids, labels, attention_mask, *model._teacher_positions(input_ids, labels, attention_mask)
    )
    assert len(kd.vecs) == 1  # _tiny_model has 2 layers; only the last is distilled
    out = model(input_ids, labels=labels, attention_mask=attention_mask)
    assert torch.isfinite(out.kd_loss)


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

    # Two token-13 markers (idx 1, 5); the masked-prompt one at idx 1 is dropped.
    batch_idx, teacher_pos = model._teacher_positions(input_ids, labels, attention_mask)
    assert batch_idx.tolist() == [0]
    assert teacher_pos.tolist() == [5]


def test_token_budget_sampler_allows_variable_rows_across_dp_ranks() -> None:
    lengths = [10, 10, 10, 10, 100, 100, 100, 100]
    samplers = [
        MegabatchSortishSampler(
            lengths,
            batch_size=4,
            megabatch_mult=2,
            max_batch_tokens=200,
            num_replicas=4,
            rank=rank,
            shuffle=False,
        )
        for rank in range(4)
    ]

    batches_by_rank = [list(sampler) for sampler in samplers]
    assert [len(batches) for batches in batches_by_rank] == [1, 1, 1, 1]
    assert [len(batches[0]) for batches in batches_by_rank] == [2, 2, 4, 2]


def test_token_budget_sampler_len_tracks_current_epoch() -> None:
    lengths = [100, 100, 100, 100, 10, 10, 10, 10, 70, 70, 20, 20]
    sampler = MegabatchSortishSampler(
        lengths,
        batch_size=4,
        megabatch_mult=2,
        max_batch_tokens=200,
        shuffle=True,
        seed=2,
    )

    assert len(sampler) == 5
    sampler.set_epoch(4)
    assert len(sampler) == 6
