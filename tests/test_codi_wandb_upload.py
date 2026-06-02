import os
import sys
import time
import uuid
from types import SimpleNamespace

import pytest
import torch
import wandb
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


def test_codi_logs_metrics_to_active_wandb_run(monkeypatch) -> None:
    torch.manual_seed(0)
    teacher = _tiny_model()
    student = _tiny_model()
    student.load_state_dict(teacher.state_dict())

    config = CodiConfig(
        line_sep_token_id=10,
        sot_token_id=11,
        eot_token_id=12,
        action_sep_token_id=13,
        latent_steps=1,
        expected_action_next_token_id=9,
        wandb_log=True,
        wandb_log_prefix="train/codi",
    )
    model = CodiModel(teacher=teacher, student=student, config=config)

    logged = []

    def log(payload, step=None):
        logged.append((payload, step))

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(run=object(), log=log),
    )

    input_ids = torch.tensor([[2, 5, 10, 7, 13, 9, 3, 0]])
    attention_mask = input_ids.ne(0).long()
    labels = input_ids.masked_fill(~attention_mask.bool(), config.ignore_index)

    output = model(
        input_ids,
        labels=labels,
        attention_mask=attention_mask,
        wandb_step=7,
    )

    assert len(logged) == 1
    payload, step = logged[0]
    assert step == 7
    assert set(payload) == {
        "train/codi/loss",
        "train/codi/lm_loss",
        "train/codi/kd_loss",
        "train/codi/num_kd_positions",
    }
    assert payload["train/codi/loss"] == output.metrics["loss"].float().item()
    assert payload["train/codi/lm_loss"] == output.metrics["lm_loss"].float().item()
    assert payload["train/codi/kd_loss"] == output.metrics["kd_loss"].float().item()
    assert (
        payload["train/codi/num_kd_positions"]
        == output.metrics["num_kd_positions"].float().item()
    )


@pytest.mark.skipif(
    os.environ.get("WANDB_UPLOAD_TEST") != "1",
    reason="set WANDB_UPLOAD_TEST=1 to create a real wandb run",
)
def test_codi_metrics_upload_to_wandb() -> None:
    project = os.environ.get("WANDB_PROJECT", "codi-wandb-upload-test")
    entity = os.environ.get("WANDB_ENTITY") or None
    sentinel = f"codi-upload-{uuid.uuid4()}"

    torch.manual_seed(0)
    teacher = _tiny_model()
    student = _tiny_model()
    student.load_state_dict(teacher.state_dict())

    config = CodiConfig(
        line_sep_token_id=10,
        sot_token_id=11,
        eot_token_id=12,
        action_sep_token_id=13,
        latent_steps=1,
        expected_action_next_token_id=9,
        wandb_log=True,
        wandb_log_prefix="upload_test/codi",
    )
    model = CodiModel(teacher=teacher, student=student, config=config)

    run = wandb.init(
        project=project,
        entity=entity,
        name=sentinel,
        mode="online",
        dir=os.environ.get("WANDB_DIR", "/tmp"),
        tags=["codi-upload-test"],
        config={"test": "test_codi_metrics_upload_to_wandb"},
    )
    assert run is not None
    run_path = "/".join(run.path) if isinstance(run.path, (list, tuple)) else run.path

    input_ids = torch.tensor([[2, 5, 10, 7, 13, 9, 3, 0]])
    attention_mask = input_ids.ne(0).long()
    labels = input_ids.masked_fill(~attention_mask.bool(), config.ignore_index)

    try:
        output = model(
            input_ids,
            labels=labels,
            attention_mask=attention_mask,
            wandb_step=0,
        )
        wandb.log({"upload_test/sentinel": sentinel}, step=0)
    finally:
        wandb.finish()

    expected_loss = output.metrics["loss"].float().item()
    deadline = time.time() + int(os.environ.get("WANDB_UPLOAD_TIMEOUT", "90"))
    api = wandb.Api()

    while time.time() < deadline:
        uploaded_run = api.run(run_path)
        rows = list(
            uploaded_run.scan_history(
                keys=[
                    "upload_test/sentinel",
                    "upload_test/codi/loss",
                    "upload_test/codi/num_kd_positions",
                ],
                page_size=100,
            )
        )
        for row in rows:
            if row.get("upload_test/sentinel") != sentinel:
                continue
            assert row["upload_test/codi/loss"] == pytest.approx(expected_loss)
            assert row["upload_test/codi/num_kd_positions"] == pytest.approx(1.0)
            return
        time.sleep(5)

    pytest.fail(f"wandb run did not expose uploaded metrics in time: {run_path}")
