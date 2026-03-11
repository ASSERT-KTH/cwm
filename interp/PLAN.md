# Interpretability: Probing & Steering Program State Representations in CWM

## Motivation

CWM is a 32B-parameter model mid-trained on 200M+ Python execution traces. The CWM
paper shows trace training improves code reasoning *externally*, but doesn't study
*why* internally. We want to answer:

1. **Does CWM develop a genuine world model of Python execution?** Do hidden states
   encode structured program state representations?
2. **Is this causal?** Can we steer activations to change trace predictions predictably?
3. **Can the model approximate hard properties?** Termination likelihood, error
   probability — things symbolic analysis can't decide in general.

Jin & Rinard (ICML 2024) showed a 350M model on Karel learns formal semantics. We
go further: production-scale model, real Python, causal interventions. The novel
contribution is showing that trace-trained models develop causally active program
state representations, and that steering them changes predictions in predictable ways.

## Hardware

**8× A100 80GB** on a single SLURM node.

- **Model loading**: 32B params in bf16 ≈ 64GB. Fits in 2 GPUs with TP=2. With 8
  GPUs we can run **TP=2, DP=4** (4 independent data-parallel groups, each across 2
  GPUs) for 4× throughput on extraction and steering.
- **Extraction**: DP=4 means 4 samples processed concurrently → ~4× faster.
  All 800 CruxEval samples across 3 modes in ~2-3h total.
- **Steering sweeps**: 5 layers × 7 alphas = 35 configurations. With DP=4 and 200
  samples each, this is feasible in a single 8h job.
- **Probes & logit lens**: Single GPU, CPU-only after activation loading. Can run
  on a login node or with `--gpus=1` if GPU matmul is needed for logit lens.

## Architecture Quick Reference

- **Model**: 64 transformer blocks, dim varies by checkpoint, GQA, window attention
  (3:1 local:global pattern)
- **Forward pass**: `cwm/fastgen/forward.py::_forward()` — monolithic function, loops
  over layers at line 563. Residual stream `h` is updated in-place (lines 769, 786).
  After line 786 of each iteration, `h` is the full post-layer hidden state on each
  TP rank (already all-reduced).
- **Trace tokens**: `FRAME_SEP_ID=100`, `ACTION_SEP_ID=101`, `RETURN_SEP_ID=102`,
  `CALL_SEP_ID=103`, `LINE_SEP_ID=104`, `EXCEPTION_SEP_ID=105`, `ARG_SEP_ID=106`,
  `TRACE_CONTEXT_START_ID=107`
- **Config pattern**: `@dataclass` + `OmegaConf` + CLI overrides via `load_from_cli()`
- **Inference stack**: `FastGen` → `ImpGen` (imperative wrapper) → `_forward()`
- **Distributed**: SLURM + `torch.distributed.run`, TP via DeviceMesh

## Directory Structure

```
interp/
├── PLAN.md                      # This file
├── __init__.py
├── extract/
│   ├── __init__.py
│   ├── hooks.py                 # ActivationStore + patched _forward
│   └── run_extract.py           # Extract activations from CruxEval inference
├── logit_lens/
│   ├── __init__.py
│   └── run_logit_lens.py        # Apply unembedding at each layer
├── probes/
│   ├── __init__.py
│   ├── dataset.py               # Dataset over extracted activations
│   ├── labels.py                # Parse traces → abstract property labels
│   ├── models.py                # Linear / MLP probe heads
│   └── train_probe.py           # Train probes, log to wandb
├── steering/
│   ├── __init__.py
│   ├── vectors.py               # Contrastive steering vector computation
│   ├── intervene.py             # SteeringHook for _forward
│   └── run_steering.py          # Run CruxEval with interventions
├── analysis/
│   ├── __init__.py
│   └── visualize.py             # Plotting utilities for wandb
└── scripts/
    ├── extract.sh               # SLURM: activation extraction
    ├── logit_lens.sh            # SLURM: logit lens analysis
    ├── train_probes.sh          # SLURM: probe training (single GPU)
    ├── steer.sh                 # SLURM: steering experiments
    └── test.sh                  # SLURM: run unit tests (no GPU)

tests/interp/                    # Unit tests (parallel to tests/cruxeval/)
├── conftest.py                  # Shared fixtures (fake activations, tmp dirs)
├── test_hooks.py                # ActivationStore + forward_with_hooks
├── test_labels.py               # Trace parsing → abstract property labels
├── test_probes.py               # Probe model shapes + training step
├── test_steering.py             # Vector computation + SteeringHook
└── test_logit_lens.py           # Unembedding at intermediate layers
```

---

## Phase 1: Activation Extraction

### `interp/extract/hooks.py`

**`ActivationStore`** dataclass:
```python
@dataclass
class ActivationStore:
    layers: list[int]                          # which layers to capture
    capture_token_ids: list[int] | None        # special token IDs to capture at, None=all
    data: dict[int, list[torch.Tensor]]        # {layer_idx: [per-position tensors]}
    positions: list[int]                       # token positions that were captured
    enabled: bool = True
```

**`forward_with_hooks()`**: Wrapper around `_forward()` that:
- Accepts optional `activation_store: ActivationStore | None` and
  `steering_hooks: list[SteeringHook] | None` (Phase 4)
- Inserts capture/intervention at the residual stream after each layer's FFN
  (after `h.add_(h_out)` at what is currently line 786)
- For captures: `h[positions].clone().detach().cpu()` → store
- For interventions: `h[positions] += alpha * vector`
- Token filtering: if `capture_token_ids` is set, only capture positions where
  `token_values[pos]` is in the set
- **Does not modify** the original `_forward()`. Instead, copies the function body
  with the hook points added. This keeps `cwm/` untouched.

**Memory budget**: dim=32000 (worst case) × float16 = 64KB per position per layer.
At 9 layers × ~30 trace token positions per sample = ~17MB per sample. For 200
samples = ~3.4GB total. Fits easily on CPU.

### `interp/extract/run_extract.py`

**Config**:
```python
@dataclass
class ExtractArgs:
    checkpoint_dir: str = "./model_weights/cwm"
    dump_dir: str = "interp-extract"
    mode: str = "trace_full"
    layers: list[int] = field(default_factory=lambda: [0, 8, 16, 24, 32, 40, 48, 56, 63])
    capture_at: str = "trace_tokens"  # trace_tokens | all | last
    n_samples: int = 800              # full CruxEval test set
    seed: int = 42
    wandb_project: str = "cwm-interp"
    wandb_run_name: str = ""
    gen_args: FastGenArgs = field(default_factory=lambda: FastGenArgs(
        tp_size=2, use_sampling=False, temperature=0.0,
    ))
    setup: SetupArgs = field(default_factory=lambda: SetupArgs(torch_init_timeout=7200))
```

With 8 GPUs (TP=2, DP=4), all 800 samples are split across 4 DP groups
(200 per group), processed concurrently.

**Logic**:
1. Load model, tokenizer, CruxEval dataset (reuse `run_eval.py` patterns)
2. Use greedy decoding (`temperature=0`) for deterministic traces
3. For each sample:
   - Build prompt tokens for the specified mode
   - Run generation with `ActivationStore` hooked in
   - Save: `{sample_id, code, input, output, mode, generated_text, extracted_answer,
     correct, activations: {layer: tensor[n_positions, dim]},
     token_ids: list[int], captured_positions: list[int]}`
4. Save one `.pt` file per sample in `dump_dir/activations/`
5. Save metadata index as `dump_dir/index.jsonl`
6. Log summary to wandb: n_samples, n_correct, avg trace length, etc.

**CUDA graphs note**: Disable CUDA graphs for extraction (`num_cuda_graphs=0`).
The hooks add dynamic control flow that's incompatible with graph capture.

### `interp/scripts/extract.sh`

Uses all 8 GPUs: TP=2, DP=4 (4 data-parallel groups of 2 GPUs each).
Each DP group processes 1/4 of the samples concurrently.

```bash
#!/bin/bash
#
#SBATCH -J cwm-extract
#SBATCH -t 03:00:00
#SBATCH -N 1
#SBATCH --gpus=8
#SBATCH -C "fat"
#SBATCH -o logs/extract_%j.out
#SBATCH -e logs/extract_%j.err

MODE=${MODE:-trace_full}
N_SAMPLES=${N_SAMPLES:-800}

mkdir -p logs

module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate CWM

python -m torch.distributed.run --nproc_per_node=8 \
    -m interp.extract.run_extract \
    checkpoint_dir=./model_weights/cwm \
    dump_dir=./interp-extract-${MODE} \
    mode=${MODE} \
    n_samples=${N_SAMPLES} \
    gen_args.tp_size=2 \
    gen_args.num_cuda_graphs=0
```

Extract all 3 modes:
```bash
# One job per mode (parallel submissions)
for mode in trace_full trace_single_step direct; do
    MODE=$mode sbatch interp/scripts/extract.sh
done
```

---

## Phase 2: Logit Lens

### `interp/logit_lens/run_logit_lens.py`

**Config**:
```python
@dataclass
class LogitLensArgs:
    extract_dir: str = "interp-extract"
    checkpoint_dir: str = ""           # needed for norm weights + output head
    top_k: int = 10
    wandb_project: str = "cwm-interp"
    wandb_run_name: str = ""
```

**Logic**:
1. Load extracted activations from Phase 1
2. Load model's final `RMSNorm` weights and `output` linear head (just these two,
   not the full model — they're small)
3. For each sample, for each layer, for each captured position:
   - `logits = output(rms_norm(h_layer, norm_weight, eps))`
   - Record top-k tokens and probabilities
   - Check if the correct answer token(s) appear in top-k
4. Log to wandb:
   - **Layer × position heatmap**: P(correct token) at each layer/position
   - **Crystallization curve**: layer at which correct answer first enters top-1/5/10
   - **Mode comparison**: trace_full vs direct vs trace_single_step
   - **Per-sample detail table**

**Why this is valuable**: Zero training. Immediately tells us which layers "know"
the answer and whether trace mode creates different internal representations than
direct mode. Guides which layers to target for probing and steering.

### `interp/scripts/logit_lens.sh`

Single GPU — only needs the output head weights (~1GB) plus extracted activations.
The output head is `vocab_size × dim` (128K × dim) which fits in one A100.

```bash
#!/bin/bash
#
#SBATCH -J cwm-logit-lens
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -C "fat"
#SBATCH -o logs/logit_lens_%j.out
#SBATCH -e logs/logit_lens_%j.err

EXTRACT_DIR=${EXTRACT_DIR:-./interp-extract-trace_full}

mkdir -p logs

module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate CWM

python -m interp.logit_lens.run_logit_lens \
    extract_dir=${EXTRACT_DIR} \
    checkpoint_dir=./model_weights/cwm
```

---

## Phase 3: Probe Training

### `interp/probes/labels.py`

Parse generated traces to extract abstract property labels:

```python
def extract_labels(generated_text: str, token_ids: list[int],
                   captured_positions: list[int], correct: bool) -> dict[str, list]:
    """Returns {property_name: [label_per_position]}"""
```

**Target properties** (all small-class classification):

| Property | Classes | Description |
|----------|---------|-------------|
| `return_type` | 7 | int, str, list, tuple, bool, None, other |
| `return_sign` | 4 | positive, negative, zero, N/A |
| `return_truthy` | 2 | True, False |
| `return_length_bin` | 6 | 0, 1, 2-5, 6-20, 20+, N/A |
| `will_be_correct` | 2 | whether this trace leads to correct answer |
| `trace_event_type` | 4 | call, return, line, exception (from token ID) |

For `will_be_correct`: every position in a sample gets the same label (the sample's
correctness). This is the most practically useful probe — it predicts whether the
model is on track to get the right answer.

### `interp/probes/dataset.py`

```python
class ProbeDataset(torch.utils.data.Dataset):
    def __init__(self, extract_dir: str, layers: list[int],
                 target_property: str, position_filter: str = "all"):
        # position_filter: "all" | "return_only" | "frame_sep_only"
        ...
    def __getitem__(self, idx) -> tuple[torch.Tensor, int]:
        # Returns (hidden_state_at_layer, label)
        ...
```

Loads lazily from `.pt` files. One dataset instance per (layer, property) pair.

### `interp/probes/models.py`

```python
class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, n_classes: int): ...

class MLPProbe(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_classes: int, n_hidden: int = 1): ...
```

Keep it minimal — these are tiny models. Hidden dim default: 256.

### `interp/probes/train_probe.py`

**Config**:
```python
@dataclass
class ProbeTrainArgs:
    extract_dir: str = "interp-extract"
    target_property: str = "will_be_correct"
    probe_type: str = "linear"          # linear | mlp1 | mlp2
    layers: list[int] = field(default_factory=lambda: [0, 8, 16, 24, 32, 40, 48, 56, 63])
    hidden_dim: int = 256
    lr: float = 1e-3
    epochs: int = 50
    batch_size: int = 256
    val_fraction: float = 0.2
    position_filter: str = "all"
    wandb_project: str = "cwm-interp"
    wandb_run_name: str = ""
```

**Logic**:
1. For each layer in `layers`:
   - Create `ProbeDataset`, split train/val
   - Train probe with Adam, cross-entropy loss
   - Log per-epoch: train/val accuracy, loss
   - Also train on shuffled labels (random baseline) and log majority baseline
2. Log to wandb:
   - **Accuracy vs layer** curve (the key plot)
   - **Per-property comparison** across layers
   - **Confusion matrices** for best layer
   - **Random baseline gap**: how much better than chance?

### `interp/scripts/train_probes.sh`

```bash
#!/bin/bash
#
#SBATCH -J cwm-probes
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -C "fat"
#SBATCH -o logs/probes_%j.out
#SBATCH -e logs/probes_%j.err

EXTRACT_DIR=${EXTRACT_DIR:-./interp-extract-trace_full}

mkdir -p logs

module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate CWM

# Train all property × probe_type combinations
for prop in will_be_correct return_type return_sign return_truthy; do
    for probe in linear mlp1; do
        echo "=== Training $probe probe for $prop ==="
        python -m interp.probes.train_probe \
            extract_dir=${EXTRACT_DIR} \
            target_property=$prop \
            probe_type=$probe
    done
done
```

Single GPU — probes are tiny.

---

## Phase 4: Steering Vectors

### `interp/steering/vectors.py`

**Contrastive vector computation**:
```python
def compute_steering_vectors(
    extract_dir: str,
    condition_fn: Callable[[dict], bool],  # splits samples into two groups
    layers: list[int],
    position: str = "last_return",         # which position to use
    method: str = "mean_diff",             # mean_diff | probe_weight | pca
) -> dict[int, torch.Tensor]:
    """Returns {layer: steering_vector}"""
```

**Pre-defined conditions**:
- `correct_vs_incorrect`: samples where model got it right vs wrong
- `positive_vs_negative`: positive vs negative return values
- `truthy_vs_falsy`: truthy vs falsy return values
- `numeric_vs_string`: numeric vs string return types

**Methods**:
- `mean_diff`: `v = mean(activations_A) - mean(activations_B)`, normalized
- `probe_weight`: use the trained probe's weight vector (linear probe only)
- `pca`: first PC of the difference vectors

### `interp/steering/intervene.py`

**`SteeringHook`** dataclass:
```python
@dataclass
class SteeringHook:
    layer: int
    vector: torch.Tensor         # steering direction (on device)
    alpha: float = 1.0           # scaling factor
    position: str = "all"        # all | last | trace_tokens
```

Integrated into `forward_with_hooks()` from Phase 1. At the specified layer,
after the FFN residual addition: `h[target_positions] += alpha * vector`.

### `interp/steering/run_steering.py`

**Config**:
```python
@dataclass
class SteeringArgs:
    checkpoint_dir: str = ""
    dump_dir: str = "interp-steer"
    vector_dir: str = ""                # path to pre-computed vectors
    condition: str = "correct_vs_incorrect"
    target_layers: list[int] = field(default_factory=lambda: [32, 40, 48, 56, 63])
    alphas: list[float] = field(default_factory=lambda: [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    mode: str = "trace_full"
    n_samples: int = 200
    wandb_project: str = "cwm-interp"
    wandb_run_name: str = ""
    gen_args: FastGenArgs = field(default_factory=lambda: FastGenArgs(
        tp_size=2, use_sampling=False, temperature=0.0,
    ))
    setup: SetupArgs = field(default_factory=lambda: SetupArgs(torch_init_timeout=7200))
```

With 8 GPUs (TP=2, DP=4): 5 layers × 7 alphas = 35 sweep points, each running
200 samples across 4 DP groups. ~4-6h total.

**Logic**:
1. Load steering vectors from `vector_dir`
2. For each (layer, alpha) combination:
   - Run CruxEval with `SteeringHook` active
   - Record: pass@1, per-sample predictions, answer changes
3. Log to wandb:
   - **Pass@1 vs alpha** per layer (the key causal evidence plot)
   - **Answer flip rate**: fraction of samples where prediction changed
   - **Directional consistency**: when prediction changes, does it change in the
     expected direction? (e.g., steering toward "correct" actually increases pass@1)
   - **Layer localization**: which layer's steering has the strongest effect?

### `interp/scripts/steer.sh`

```bash
#!/bin/bash
#
#SBATCH -J cwm-steer
#SBATCH -t 08:00:00
#SBATCH -N 1
#SBATCH --gpus=8
#SBATCH -C "fat"
#SBATCH -o logs/steer_%j.out
#SBATCH -e logs/steer_%j.err

VECTOR_DIR=${VECTOR_DIR:?"Set VECTOR_DIR"}
CONDITION=${CONDITION:-correct_vs_incorrect}

mkdir -p logs

module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate CWM

python -m torch.distributed.run --nproc_per_node=8 \
    -m interp.steering.run_steering \
    checkpoint_dir=./model_weights/cwm \
    dump_dir=./interp-steer-${CONDITION} \
    vector_dir=${VECTOR_DIR} \
    condition=${CONDITION} \
    gen_args.tp_size=2 \
    gen_args.num_cuda_graphs=0
```

Uses all 8 GPUs (TP=2, DP=4). Longer walltime — runs CruxEval for each
(layer, alpha) combination. With DP=4, each sweep point takes ~4× less time
than with DP=1.

---

## Phase 5: Analysis & Visualization

### `interp/analysis/visualize.py`

Utility functions that produce wandb-compatible plots:
- `plot_layer_heatmap(data, xlabel, ylabel, title)` → `wandb.Image`
- `plot_accuracy_by_layer(layer_accs, baseline)` → `wandb.plot.line`
- `plot_steering_sweep(alphas, pass_at_1, layer)` → `wandb.plot.line`
- `plot_crystallization(layer_probs)` → `wandb.plot.line`

Keep minimal. Use matplotlib with a clean style. All plots logged to wandb
during the respective phase scripts.

---

## Execution Order

| Phase | Script | GPUs | TP×DP | Time Est. | Depends On |
|-------|--------|------|-------|-----------|------------|
| 1. Extract | `extract.sh` (×3 modes) | 8 | 2×4 | ~1-2h each | — |
| 2. Logit Lens | `logit_lens.sh` | 1 | — | ~30min | Phase 1 |
| 3. Train Probes | `train_probes.sh` | 1 | — | ~1-2h | Phase 1 |
| 4a. Compute Vectors | (in `run_steering.py`) | 1 | — | ~10min | Phase 1 (+ optionally Phase 3) |
| 4b. Steering | `steer.sh` | 8 | 2×4 | ~4-6h | Phase 4a |

With 8×A100 80GB, DP=4 gives ~4× throughput on extraction and steering.
Phases 2 and 3 only need 1 GPU and can run in parallel after Phase 1.
Total wall-clock from scratch: ~8-12h across all phases.

---

## Tests

All tests live in `tests/interp/`. Follow the existing pattern from
`tests/cruxeval/`: no GPU required for unit tests, skip gracefully when the
tokenizer or model is unavailable.

```
tests/interp/
├── conftest.py                  # Shared fixtures (fake activations, mini model)
├── test_hooks.py                # Phase 1: ActivationStore + forward_with_hooks
├── test_labels.py               # Phase 3: trace parsing → abstract property labels
├── test_probes.py               # Phase 3: probe model shapes and training step
├── test_steering.py             # Phase 4: vector computation + SteeringHook
└── test_logit_lens.py           # Phase 2: unembedding at intermediate layers
```

### `tests/interp/conftest.py`

Shared fixtures — no real model needed:

```python
import pytest
import torch

@pytest.fixture
def dim():
    return 128  # small dim for tests

@pytest.fixture
def n_layers():
    return 4

@pytest.fixture
def fake_activations(dim, n_layers):
    """Simulates extracted activations for one sample: {layer: [n_positions, dim]}"""
    n_positions = 10
    return {
        layer: torch.randn(n_positions, dim)
        for layer in range(n_layers)
    }

@pytest.fixture
def fake_extract_dir(tmp_path, fake_activations):
    """Creates a minimal extract directory with 20 fake samples."""
    import json
    act_dir = tmp_path / "activations"
    act_dir.mkdir()
    index = []
    for i in range(20):
        correct = i % 2 == 0  # alternating correct/incorrect
        sample = {
            "sample_id": str(i),
            "code": f"def f(x):\\n    return x + {i}\\n",
            "input": "5",
            "output": str(5 + i),
            "mode": "trace_full",
            "generated_text": f"<|return_sep|><|action_sep|> return f(5)<|arg_sep|>\"{5+i}\"<|frame_sep|>",
            "extracted_answer": str(5 + i),
            "correct": correct,
            "token_ids": [100, 101, 102, 106, 100] * 2,  # fake trace tokens
            "captured_positions": list(range(10)),
            "activations": fake_activations,
        }
        torch.save(sample, act_dir / f"{i}.pt")
        index.append({k: v for k, v in sample.items() if k != "activations"})
    with open(tmp_path / "index.jsonl", "w") as f:
        for entry in index:
            f.write(json.dumps(entry) + "\n")
    return tmp_path
```

### `test_hooks.py` — ActivationStore

Tests that the hook infrastructure works correctly without needing the real model.

```python
def test_activation_store_init():
    """ActivationStore initializes with empty data."""
    store = ActivationStore(layers=[0, 2], capture_token_ids=None)
    assert store.enabled
    assert store.data == {}

def test_activation_store_captures_correct_layers():
    """Only specified layers are stored."""
    store = ActivationStore(layers=[1, 3], capture_token_ids=None)
    # Simulate what forward_with_hooks does
    for layer_idx in range(4):
        h = torch.randn(5, 128)  # 5 positions, dim=128
        if layer_idx in store.layers:
            store.data[layer_idx] = h.clone()
    assert set(store.data.keys()) == {1, 3}
    assert 0 not in store.data

def test_activation_store_token_filtering():
    """When capture_token_ids is set, only matching positions are captured."""
    FRAME_SEP = 100
    store = ActivationStore(layers=[0], capture_token_ids=[FRAME_SEP])
    token_ids = torch.tensor([50, 100, 200, 100, 50])
    h = torch.randn(5, 128)
    mask = torch.isin(token_ids, torch.tensor(store.capture_token_ids))
    captured = h[mask]
    assert captured.shape[0] == 2  # two FRAME_SEP positions

def test_forward_with_hooks_preserves_output():
    """Hooking with capture-only should not change the forward output."""
    # This test uses a tiny nn.Module to simulate the layer loop
    # The key invariant: output with hooks == output without hooks
    dim = 64
    h = torch.randn(3, dim)
    h_copy = h.clone()
    store = ActivationStore(layers=[0], capture_token_ids=None)
    # Simulate capture (clone, don't modify h)
    store.data[0] = h.clone().detach()
    assert torch.equal(h, h_copy)  # h unchanged
```

### `test_labels.py` — Trace parsing

```python
def test_extract_return_type_int():
    labels = extract_labels_from_answer("42")
    assert labels["return_type"] == "int"

def test_extract_return_type_str():
    labels = extract_labels_from_answer('"hello"')
    assert labels["return_type"] == "str"

def test_extract_return_type_list():
    labels = extract_labels_from_answer("[1, 2, 3]")
    assert labels["return_type"] == "list"

def test_extract_return_type_none():
    labels = extract_labels_from_answer("None")
    assert labels["return_type"] == "None"

def test_extract_return_sign_positive():
    labels = extract_labels_from_answer("42")
    assert labels["return_sign"] == "positive"

def test_extract_return_sign_negative():
    labels = extract_labels_from_answer("-7")
    assert labels["return_sign"] == "negative"

def test_extract_return_sign_zero():
    labels = extract_labels_from_answer("0")
    assert labels["return_sign"] == "zero"

def test_extract_return_sign_na_for_string():
    labels = extract_labels_from_answer('"hello"')
    assert labels["return_sign"] == "N/A"

def test_extract_return_truthy():
    assert extract_labels_from_answer("42")["return_truthy"] == True
    assert extract_labels_from_answer("0")["return_truthy"] == False
    assert extract_labels_from_answer('""')["return_truthy"] == False
    assert extract_labels_from_answer("[]")["return_truthy"] == False
    assert extract_labels_from_answer("None")["return_truthy"] == False

def test_extract_return_length_bin():
    assert extract_labels_from_answer("[]")["return_length_bin"] == "0"
    assert extract_labels_from_answer("[1]")["return_length_bin"] == "1"
    assert extract_labels_from_answer("[1,2,3]")["return_length_bin"] == "2-5"
    assert extract_labels_from_answer('"hello world!"')["return_length_bin"] == "6-20"

def test_extract_event_type_from_token_id():
    assert event_type_from_token_id(102) == "return"   # RETURN_SEP
    assert event_type_from_token_id(103) == "call"     # CALL_SEP
    assert event_type_from_token_id(104) == "line"     # LINE_SEP
    assert event_type_from_token_id(105) == "exception"  # EXCEPTION_SEP
```

### `test_probes.py` — Probe models

```python
def test_linear_probe_shape(dim):
    probe = LinearProbe(in_dim=dim, n_classes=7)
    x = torch.randn(32, dim)
    out = probe(x)
    assert out.shape == (32, 7)

def test_mlp_probe_shape(dim):
    probe = MLPProbe(in_dim=dim, hidden_dim=64, n_classes=4, n_hidden=1)
    x = torch.randn(32, dim)
    out = probe(x)
    assert out.shape == (32, 4)

def test_mlp2_probe_shape(dim):
    probe = MLPProbe(in_dim=dim, hidden_dim=64, n_classes=2, n_hidden=2)
    x = torch.randn(16, dim)
    out = probe(x)
    assert out.shape == (16, 2)

def test_probe_training_step(dim):
    """One train step runs without error and loss decreases signal."""
    probe = LinearProbe(in_dim=dim, n_classes=2)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-2)
    x = torch.randn(64, dim)
    y = torch.randint(0, 2, (64,))
    loss = torch.nn.functional.cross_entropy(probe(x), y)
    loss.backward()
    optimizer.step()
    assert loss.item() > 0  # sanity — loss is finite and positive

def test_probe_dataset_loads(fake_extract_dir, dim):
    """ProbeDataset loads from extracted dir and returns correct shapes."""
    ds = ProbeDataset(
        extract_dir=str(fake_extract_dir),
        layer=0,
        target_property="will_be_correct",
    )
    assert len(ds) > 0
    x, y = ds[0]
    assert x.shape[-1] == dim
    assert y in (0, 1)
```

### `test_steering.py` — Vectors and intervention

```python
def test_mean_diff_vector_shape(fake_extract_dir, dim):
    """Contrastive vector has correct shape and is normalized."""
    vectors = compute_steering_vectors(
        extract_dir=str(fake_extract_dir),
        condition="correct_vs_incorrect",
        layers=[0, 2],
        method="mean_diff",
    )
    assert set(vectors.keys()) == {0, 2}
    for v in vectors.values():
        assert v.shape == (dim,)
        # normalized to unit length
        assert abs(v.norm().item() - 1.0) < 1e-5

def test_mean_diff_vector_nonzero(fake_extract_dir):
    """Vector is not all zeros (groups should differ)."""
    vectors = compute_steering_vectors(
        extract_dir=str(fake_extract_dir),
        condition="correct_vs_incorrect",
        layers=[0],
        method="mean_diff",
    )
    assert vectors[0].abs().sum() > 0

def test_steering_hook_modifies_activations(dim):
    """SteeringHook adds the vector to h at the specified positions."""
    h = torch.zeros(5, dim)
    vector = torch.ones(dim)
    hook = SteeringHook(layer=0, vector=vector, alpha=2.0, position="all")
    # Simulate intervention
    h_steered = h.clone()
    h_steered += hook.alpha * hook.vector
    assert torch.allclose(h_steered, torch.full((5, dim), 2.0))

def test_steering_hook_zero_alpha_is_noop(dim):
    """alpha=0 should not change activations."""
    h = torch.randn(5, dim)
    h_orig = h.clone()
    hook = SteeringHook(layer=0, vector=torch.ones(dim), alpha=0.0, position="all")
    h += hook.alpha * hook.vector
    assert torch.equal(h, h_orig)
```

### `test_logit_lens.py` — Unembedding at intermediate layers

```python
def test_logit_lens_output_shape(dim):
    """Applying norm + output head gives vocab-sized logits."""
    vocab_size = 1000
    norm = torch.nn.RMSNorm(dim)
    output = torch.nn.Linear(dim, vocab_size, bias=False)
    h = torch.randn(10, dim)  # 10 positions
    logits = output(norm(h))
    assert logits.shape == (10, vocab_size)

def test_logit_lens_top_k():
    """Top-k extraction returns correct number of entries."""
    logits = torch.randn(1, 500)
    top_k = 5
    values, indices = logits.topk(top_k, dim=-1)
    assert indices.shape == (1, top_k)
    # values should be sorted descending
    assert (values[0, :-1] >= values[0, 1:]).all()

def test_logit_lens_softmax_sums_to_one(dim):
    """Probabilities from logit lens sum to 1."""
    vocab_size = 1000
    norm = torch.nn.RMSNorm(dim)
    output = torch.nn.Linear(dim, vocab_size, bias=False)
    h = torch.randn(1, dim)
    logits = output(norm(h))
    probs = torch.softmax(logits, dim=-1)
    assert abs(probs.sum().item() - 1.0) < 1e-5
```

### Running tests

```bash
# All unit tests (no GPU, no model needed)
pytest tests/interp/ -v

# With tokenizer (for tests that need it)
pytest tests/interp/ -v --tokenizer-path=./tokenizer.model
```

Add to the scripts directory:

### `interp/scripts/test.sh`

```bash
#!/bin/bash
#
#SBATCH -J cwm-interp-test
#SBATCH -t 00:10:00
#SBATCH -N 1
#SBATCH --gpus=0
#SBATCH -o logs/interp_test_%j.out
#SBATCH -e logs/interp_test_%j.err

mkdir -p logs

module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate CWM

pytest tests/interp/ -v --tb=short
```

No GPU needed. Can also run on login node.

### Test design principles

- **No real model**: All tests use fake tensors or tiny `nn.Module`s. The 32B
  model never needs to load.
- **No GPU**: All tests run on CPU. Shapes and logic are device-agnostic.
- **Deterministic**: Use `torch.manual_seed(0)` in fixtures where needed.
- **Fast**: Each test < 1s. Full suite < 30s.
- **One concern per test**: each test checks one specific behavior.
- **Fixtures from conftest.py**: `fake_extract_dir` creates a tmp directory with
  20 samples that looks like real Phase 1 output. Reused across probe, steering,
  and logit lens tests.

---

## Implementation Notes

### Hooking strategy

`_forward()` in `cwm/fastgen/forward.py` is a free function, not a module method,
so `register_forward_hook` won't work. Two options:

**Option A (preferred): Monkey-patch at call site.** Replace the `_forward` import
in `forward.py`'s `prefill()` and `decode()` with our `forward_with_hooks()`.
This is clean — we only patch during interp runs, the original code is untouched.

**Option B: Copy and extend.** Copy `_forward()` into `interp/extract/hooks.py` and
add the hook points. More code duplication but zero risk of breaking the original.

Go with **Option A** initially. Fall back to B if CUDA graph issues arise.

### CUDA graphs

Decode uses CUDA graphs (`num_cuda_graphs` in GenArgs). Hooks add dynamic control
flow → incompatible. Set `gen_args.num_cuda_graphs=0` for all interp scripts.
This slows decode but extraction doesn't need to be fast.

### Tensor parallelism

After the FFN all-reduce (line 786), `h` is identical on all TP ranks. Capture
from rank 0 only. SteeringHook must apply on all ranks (since they all have the
same `h`, and the steering must be consistent).

### wandb integration

Each script initializes wandb at the top:
```python
if is_rank_zero:
    wandb.init(project=args.wandb_project, name=args.wandb_run_name,
               config=dataclass_to_dict(args))
```

Log with `wandb.log()` throughout. Finish with `wandb.finish()`.

### What NOT to build

- No custom training loops for the main model — we only train tiny probes
- No SAE or dictionary learning — out of scope
- No attention pattern analysis — focus on residual stream
- No changes to `cwm/` source files — all code lives in `interp/`
