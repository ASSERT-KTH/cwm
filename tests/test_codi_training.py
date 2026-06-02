import torch
from transformers import CwmConfig, CwmForCausalLM

from cwm.training.codi import CodiConfig, CodiModel


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
        line_sep_token_id=10,
        sot_token_id=11,
        eot_token_id=12,
        action_sep_token_id=13,
        latent_steps=2,
        expected_action_next_token_id=9,
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



def test_codi_kd_positions_ignore_masked_prompt_tokens() -> None:
    torch.manual_seed(0)
    student = _tiny_model()

    config = CodiConfig(
        line_sep_token_id=10,
        sot_token_id=11,
        eot_token_id=12,
        action_sep_token_id=13,
        latent_steps=1,
        expected_action_next_token_id=9,
    )
    model = CodiModel(student=student, config=config)

    input_ids = torch.tensor([[2, 13, 9, 10, 7, 13, 9, 3]])
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:, :4] = config.ignore_index

    output = model(input_ids, labels=labels, attention_mask=attention_mask)

    assert output.kd_positions.tolist() == [[0, 6, 9]]
