"""GPU integration tests for the interp pipeline.

Requires the real model checkpoint. Run with a single GPU (no distributed):

    python -m pytest tests/interp/test_integration_gpu.py -v \\
        --checkpoint-dir=./model_weights/cwm -s

The module-scoped fixture loads FastGen once and shares it across all tests.
Each test creates a fresh ImpGen from the shared FastGen — ImpGen is a lightweight
wrapper (no GPU allocation), so this is safe.
"""

import json
import queue
import threading
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module")
def checkpoint_dir(request):
    path = request.config.getoption("--checkpoint-dir", default=None)
    if not path:
        pytest.skip("Pass --checkpoint-dir to run GPU integration tests")
    if not Path(path).exists():
        pytest.skip(f"Checkpoint dir not found: {path}")
    return path


# ---------------------------------------------------------------------------
# Shared model fixture: returns (FastGen, tokenizer, tp_group)
# Each test creates its own ImpGen from FastGen.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def loaded_fastgen(checkpoint_dir):
    """Load FastGen with tp_size=1 on a single GPU. Shared across all tests."""
    import os
    import torch.distributed

    from cwm.common.environment import init_torch_distributed, setup_env, setup_torch_flags
    from cwm.fastgen.generate import FastGen
    from cwm.fastgen.utils.loading import build_fastgen_model, build_tokenizer_from_ckpt
    from evals.args import FastGenArgs
    from torch.distributed.device_mesh import init_device_mesh

    # Emulate torch.distributed.run single-process env so environment.py takes
    # the torch-run path (RANK/WORLD_SIZE) rather than the SLURM path.
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29600")

    setup_env(mp_spawn_method="forkserver")
    init_torch_distributed(timeout=120)
    setup_torch_flags()

    world_mesh = init_device_mesh(
        device_type="cuda",
        mesh_shape=(1, 1),
        mesh_dim_names=("dp", "tp"),
    )
    tp_group = torch.distributed.new_group([0], backend="moodist")

    tokenizer = build_tokenizer_from_ckpt(checkpoint_dir)
    model = build_fastgen_model(
        world_mesh=world_mesh,
        checkpoint_dir=checkpoint_dir,
        vocab_parallel=False,
        loss_parallel=False,
    )

    gen_args = FastGenArgs(
        tp_size=1,
        use_sampling=False,
        temperature=0.0,
        num_cuda_graphs=0,
        max_batch=1,
        # Tiny KV cache: 512 tokens × 8 kv_heads × 256 head_dim × 64 layers × bf16
        # ≈ 64 MB total — fits in the ~300 MB left after model weights load.
        # Default (166400) would need 650 MB/layer × 64 = OOM.
        max_seq=512,
    )
    fg = FastGen(
        gen_args,
        model=model,
        tokenizer=tokenizer,
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
        tp_mesh=world_mesh["tp"],
    )
    torch.cuda.empty_cache()

    yield fg, tokenizer, tp_group

    fg.destroy()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


# ---------------------------------------------------------------------------
# Helper: run a list of (prompt_tokens, gen_kwargs, context_manager) tuples
# in a single ImpGen session. Returns list of Packet.
# ---------------------------------------------------------------------------


def _run_session(fg, tp_group, jobs):
    """Run all (prompt_tokens, gen_kwargs, ctx) in one ImpGen session.

    jobs: list of (prompt_tokens, gen_kwargs, optional_context_manager)
    Returns: list of Packet in the same order.
    """
    from cwm.rl.lib.impgen import ImpGen

    g = ImpGen(fg, tp_rank=0, tp_group=tp_group)
    results = [None] * len(jobs)
    exc_q: queue.Queue = queue.Queue()
    done_event = threading.Event()

    def worker():
        try:
            for i, (prompt_tokens, gen_kwargs, ctx) in enumerate(jobs):
                if ctx is not None:
                    with ctx:
                        pkt = g.generate(tokens=prompt_tokens, **gen_kwargs)
                else:
                    pkt = g.generate(tokens=prompt_tokens, **gen_kwargs)
                results[i] = pkt
        except Exception as e:
            exc_q.put(e)
        finally:
            done_event.set()

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    while True:
        done = g.work()
        if done:
            break
        if done_event.is_set():
            g.stop()
        if not exc_q.empty():
            raise RuntimeError("Worker failed") from exc_q.get()

    t.join()
    if not exc_q.empty():
        raise RuntimeError("Worker failed") from exc_q.get()
    return results


# ---------------------------------------------------------------------------
# Test 1: hook fires during real generation
# ---------------------------------------------------------------------------


def test_hooks_fire_on_real_model(loaded_fastgen):
    """After install_forward_hooks(), store.data should be populated."""
    from interp.extract.hooks import (
        ActivationStore,
        activation_hook_context,
        install_forward_hooks,
        uninstall_forward_hooks,
    )

    fg, tokenizer, tp_group = loaded_fastgen

    install_forward_hooks()
    try:
        prompt = "def f(x):\n    return x + 1\n"
        prompt_tokens = tokenizer.encode(prompt, bos=True)

        layers_to_capture = [0, 16, 32, 48, 63]
        store = ActivationStore(layers=layers_to_capture, capture_token_ids=None)
        ctx = activation_hook_context(store=store)

        _run_session(fg, tp_group, [(prompt_tokens, {"max_gen": 10}, ctx)])

        assert store.data, "store.data is empty — hooks did not fire"
        for layer in layers_to_capture:
            assert layer in store.data, f"Layer {layer} not captured"
            act = store.get_activations(layer)
            assert act is not None
            assert act.ndim == 2
            assert act.shape[-1] == 6144, f"Expected dim=6144, got {act.shape[-1]}"
            assert act.shape[0] > 0, f"Layer {layer}: no positions captured"
    finally:
        uninstall_forward_hooks()


# ---------------------------------------------------------------------------
# Test 2: capture-at-trace-tokens filter
# ---------------------------------------------------------------------------


def test_trace_token_filtering(loaded_fastgen):
    """Captured positions should only contain trace separator tokens."""
    from interp.extract.hooks import (
        ActivationStore,
        activation_hook_context,
        install_forward_hooks,
        uninstall_forward_hooks,
    )
    from evals.cruxeval.prompts import make_trace_full_prompt_tokens

    fg, tokenizer, tp_group = loaded_fastgen

    code = "def f(x):\n    return x + 1\n"
    inp = "5"
    TRACE_TOKEN_IDS = [100, 101, 102, 103, 104, 105, 106, 107]

    install_forward_hooks()
    try:
        prompt_tokens = make_trace_full_prompt_tokens(code, inp, tokenizer)
        store = ActivationStore(layers=[32], capture_token_ids=TRACE_TOKEN_IDS)
        ctx = activation_hook_context(store=store)

        _run_session(fg, tp_group, [(prompt_tokens, {"max_gen": 64}, ctx)])

        if 32 in store.data:
            act = store.get_activations(32)
            assert act is not None
            assert act.shape[-1] == 6144
    finally:
        uninstall_forward_hooks()


# ---------------------------------------------------------------------------
# Test 3: uninstall restores original _forward
# ---------------------------------------------------------------------------


def test_uninstall_restores_original():
    """After uninstall_forward_hooks(), cwm.fastgen.forward._forward is original."""
    import cwm.fastgen.forward as fwd
    from interp.extract.hooks import (
        _hooked_forward,
        install_forward_hooks,
        uninstall_forward_hooks,
    )

    original = fwd._forward
    install_forward_hooks()
    assert fwd._forward is _hooked_forward, "Hook not installed"
    uninstall_forward_hooks()
    assert fwd._forward is original, "Original not restored"


# ---------------------------------------------------------------------------
# Test 4: end-to-end mini extraction (2 samples)
# ---------------------------------------------------------------------------


def test_mini_extraction(loaded_fastgen, tmp_path):
    """Run extraction on 2 CruxEval samples, verify output files and shapes."""
    from datasets import load_dataset
    from evals.cruxeval.evaluate import check_correct, extract_answer_trace_full
    from evals.cruxeval.prompts import make_trace_full_prompt_tokens
    from interp.extract.hooks import (
        ActivationStore,
        activation_hook_context,
        install_forward_hooks,
        uninstall_forward_hooks,
    )

    fg, tokenizer, tp_group = loaded_fastgen
    dataset = list(load_dataset("cruxeval-org/cruxeval", split="test"))[:2]

    act_dir = tmp_path / "activations"
    act_dir.mkdir()

    layers = [0, 32, 63]
    TRACE_TOKEN_IDS = [100, 101, 102, 103, 104, 105, 106, 107]

    # Build one store per sample
    stores = [
        ActivationStore(layers=layers, capture_token_ids=TRACE_TOKEN_IDS)
        for _ in dataset
    ]
    jobs = [
        (
            make_trace_full_prompt_tokens(s["code"], s["input"], tokenizer),
            {"max_gen": 64},
            activation_hook_context(store=stores[i]),
        )
        for i, s in enumerate(dataset)
    ]

    install_forward_hooks()
    try:
        packets = _run_session(fg, tp_group, jobs)
    finally:
        uninstall_forward_hooks()

    index = []
    for i, (sample, packet, store) in enumerate(zip(dataset, packets, stores)):
        generated_text = tokenizer.decode(packet.tokens, cut_at_stop_tokens=False)
        predicted = extract_answer_trace_full(generated_text, sample["input"])
        correct = (
            check_correct(sample["code"], sample["output"], predicted)
            if predicted is not None
            else False
        )
        activations = {
            layer: store.get_activations(layer)
            for layer in layers
            if store.get_activations(layer) is not None
        }
        out = {
            "sample_id": sample["id"],
            "code": sample["code"],
            "input": sample["input"],
            "output": sample["output"],
            "mode": "trace_full",
            "generated_text": generated_text,
            "extracted_answer": predicted,
            "correct": correct,
            "token_ids": packet.tokens,
            "captured_positions": store.positions,
            "activations": activations,
        }
        torch.save(out, act_dir / f"{sample['id']}.pt")
        index.append({k: v for k, v in out.items() if k != "activations"})

    with (tmp_path / "index.jsonl").open("w") as f:
        for e in index:
            f.write(json.dumps(e, default=str) + "\n")

    # --- Assertions ---
    for sample in dataset:
        pt_path = act_dir / f"{sample['id']}.pt"
        assert pt_path.exists(), f"Missing .pt for {sample['id']}"

        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        assert "activations" in data
        assert "token_ids" in data
        assert "generated_text" in data

        for layer in layers:
            if layer in data["activations"]:
                act = data["activations"][layer]
                assert act.ndim == 2, f"Layer {layer}: expected 2D tensor"
                assert act.shape[-1] == 6144, f"Layer {layer}: dim={act.shape[-1]}"

    entries = [
        json.loads(line)
        for line in (tmp_path / "index.jsonl").read_text().splitlines()
    ]
    assert len(entries) == 2
    print(f"\nMini extraction: {sum(e['correct'] for e in entries)}/2 correct")
    for e in entries:
        print(f"  [{e['sample_id']}] correct={e['correct']} answer={e['extracted_answer']!r}")


# ---------------------------------------------------------------------------
# Test 5: steering vectors from real activations
# ---------------------------------------------------------------------------


def test_steering_vectors_from_real_activations(loaded_fastgen, tmp_path):
    """Compute steering vectors from 10 samples; verify shape and unit norm."""
    from datasets import load_dataset
    from evals.cruxeval.evaluate import check_correct, extract_answer_trace_full
    from evals.cruxeval.prompts import make_trace_full_prompt_tokens
    from interp.extract.hooks import (
        ActivationStore,
        activation_hook_context,
        install_forward_hooks,
        uninstall_forward_hooks,
    )
    from interp.steering.vectors import compute_steering_vectors

    fg, tokenizer, tp_group = loaded_fastgen
    dataset = list(load_dataset("cruxeval-org/cruxeval", split="test"))[:10]

    act_dir = tmp_path / "activations"
    act_dir.mkdir()
    layers = [32]
    TRACE_TOKEN_IDS = [100, 101, 102, 103, 104, 105, 106, 107]

    stores = [
        ActivationStore(layers=layers, capture_token_ids=TRACE_TOKEN_IDS)
        for _ in dataset
    ]
    jobs = [
        (
            make_trace_full_prompt_tokens(s["code"], s["input"], tokenizer),
            {"max_gen": 64},
            activation_hook_context(store=stores[i]),
        )
        for i, s in enumerate(dataset)
    ]

    install_forward_hooks()
    try:
        packets = _run_session(fg, tp_group, jobs)
    finally:
        uninstall_forward_hooks()

    index = []
    for sample, packet, store in zip(dataset, packets, stores):
        generated_text = tokenizer.decode(packet.tokens, cut_at_stop_tokens=False)
        predicted = extract_answer_trace_full(generated_text, sample["input"])
        correct = (
            check_correct(sample["code"], sample["output"], predicted)
            if predicted is not None
            else False
        )
        activations = {
            layer: store.get_activations(layer)
            for layer in layers
            if store.get_activations(layer) is not None
        }
        out = {
            "sample_id": sample["id"],
            "correct": correct,
            "extracted_answer": predicted,
            "token_ids": packet.tokens,
            "captured_positions": store.positions,
            "activations": activations,
        }
        torch.save(out, act_dir / f"{sample['id']}.pt")
        index.append({k: v for k, v in out.items() if k != "activations"})

    with (tmp_path / "index.jsonl").open("w") as f:
        for e in index:
            f.write(json.dumps(e, default=str) + "\n")

    n_correct = sum(e["correct"] for e in index)
    print(f"\nSteering test: {n_correct}/10 correct")

    if n_correct == 0 or n_correct == len(dataset):
        pytest.skip("All samples same label — cannot compute contrastive vector")

    vectors = compute_steering_vectors(
        extract_dir=str(tmp_path),
        condition="correct_vs_incorrect",
        layers=layers,
        method="mean_diff",
    )

    assert 32 in vectors, "No vector computed for layer 32"
    v = vectors[32]
    assert v.shape == (6144,), f"Expected (6144,), got {v.shape}"
    assert abs(v.norm().item() - 1.0) < 1e-4, f"Not unit: norm={v.norm().item()}"
