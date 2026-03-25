# Plan: Tracing LLM Program Representations Token-by-Token During Bug Fixing

## Context
When CWM generates tokens on a bug-fixing task, we want to know: does the model maintain an evolving latent representation of "which program it's thinking about"? Does that representation shift from a "buggy state" toward a "fixed state" as reasoning progresses? And can we identify, decode, steer, and causally validate such a representation? We don't assume the representation exists — finding it is one of the goals.

The system runs on a SLURM cluster (Berzelius) with up to 8× A100 80GB GPUs available. GPU jobs scale from 1–8 GPUs depending on task (TP=2 or TP=4 or TP=8, DP=1 to DP=4). Analysis runs on CPU nodes. The plan is structured for autonomous iterative execution: submit jobs, analyze results, decide next experiments, repeat. All experiments must be reproducible: fixed seeds, exact commands recorded, all outputs checksummed.

---

## Core Technical Insight

At each decode step T, CWM processes exactly one new token. The forward pass produces `h_T[-1, L]` ∈ ℝ^6144 — the single hidden state of the newly generated token at layer L. This is the model's full "current state," encoding everything: the buggy program, the task description, and all reasoning tokens generated so far.

The sequence `{h_T : T=1..N}` across all decode steps is the **representation trajectory** — the model iterating, token by token, through internal states as it reasons.

---

## Dataset: CRUXEval + Controlled Mutations

Take the 800 CRUXEval programs. Apply systematic mutations:
- **Off-by-one**: `range(n)` → `range(n-1)`, index ±1
- **Wrong operator**: `+` → `-`, `>` → `>=`, `*` → `+`, `and` → `or`
- **Condition flip**: `if x:` → `if not x:`
- **Wrong comparator**: `<` → `<=`, `>` → `>=`

**Validity filter**: only keep mutations where the output changes (verified by execution with 5s timeout). Up to 3 buggy variants per original.

**Actual dataset** (`interp/bug_trace/data/pairs.json`): 830 pairs from ~450 unique originals.

| Mutation | Count |
|----------|-------|
| `condition_flip` | 288 |
| `off_by_one_minus` | 176 |
| `off_by_one_plus` | 170 |
| `wrong_comparator` | 115 |
| `wrong_operator` | 81 |
| **Total** | **830** |

**Total extraction samples**: 1280 (450 deduplicated originals + 830 buggy variants).

---

## Two Experimental Tracks

### Track A: NL Reasoning Mode (primary)
Prompt: system prompt instructing `<think>` reasoning, then user message with buggy code + wrong/correct output, assistant prefilled with `<think>\n`.

CWM generates reasoning in `<think>`, then outputs the fix. Track `h_T[-1, L]` during generation.

**Core question**: Does the trajectory in ℝ^6144 shift directionally from "buggy space" to "fix space" during reasoning?

### Track B: Execution Trace Mode
CWM traces the buggy program step by step (`trace_full` mode). Track `h_T[-1, L]` at every generated token.

**Core question**: Do representations at trace separator tokens encode the final return value? When does the model "commit" to an answer during tracing?

Both tracks use identical infrastructure. Track B is easier (CWM is expert at trace generation); Track A is the primary interest.

---

## Methods for Finding "The Representation"

### Tier 1 — Immediate

**1. Logit lens over decode steps**
At each step T and layer L, project `h_T[-1, L]` through the unembedding head. Get P(top-1 token | T, L). Find T*(L) = first step where P > threshold. Produces a 2D "commitment surface" heatmap. No labels needed.

**2. Linear probing**
Labels: `is_buggy`, `will_be_correct`. Train a linear classifier at each (layer L, relative time bin B). Produces a 2D accuracy heatmap (T × L). Finds where the representation most cleanly encodes program fate.

### Tier 2 — Second iteration

**3. PCA / Functional PCA**
Stack all trajectories `{h_T[L, :]}`. Fit PCA. Check: does PC1 separate correct/buggy? Plot trajectories in PC1-PC2 space colored by outcome.

**4. Change point detection**
For each trajectory, find T* = step with largest inter-step distance (cosine or L2). This is the "aha moment." Correlate T* with generated text at that position.

**5. CCS (Contrast-Consistent Search)** — Burns et al. 2022
For each (original, buggy) pair, find a direction `d` such that `d·h(original) = -d·h(buggy)` consistently. Label-free.
Loss: `L(d) = E[(d·h+ + d·h-)^2] + E[(d·h+ - d·h- - 1)^2]`.

### Tier 3 — Third iteration

**6. Activation patching / Causal tracing**
Steer buggy generations with the CCS direction at (T, L). Find the minimal intervention that restores the correct answer. Gives precise causal attribution.

**7. Representational Similarity Analysis (RSA)**
Compare pairwise model similarity matrix to pairwise program edit-distance matrix. High Spearman ρ → model geometry reflects program structure.

**8. Dynamic Mode Decomposition (DMD)**
Fit `h_{T+1} ≈ A · h_T`. Eigenvalues of A reveal dominant temporal modes. Persistent modes (|λ| ≈ 1) are "working memory" candidates.

**9. Sparse Autoencoders (SAEs)** — future work
Train `h ≈ W·f + b` with L1 penalty on `f`. Each active feature may correspond to a specific bug type or program property.

---

## Autonomous Iteration Loop

```
ITERATION 1: Build dataset + run Tier 1 analysis
  → probe_accuracy(is_buggy) < 0.6 at ALL layers → try Track B
  → probe_accuracy(will_be_correct) ≥ 0.6 → representation exists → Tier 2

ITERATION 2: Tier 2 analysis + causal validation
  → change_point T* correlates with text ("The bug is...") → strong finding → steer
  → T* doesn't correlate → try activation patching

ITERATION 3: Tier 3 methods
  → Activation patching: find critical (T*, L*)
  → RSA: measure program geometry
  → DMD: find persistent modes

ITERATION 4+: Refine and extend
  → Out-of-distribution bugs, harder programs, different prompting, SAEs
```

**Decision thresholds:**
- `probe_accuracy(is_buggy) < 0.6` at ALL layers → switch to Track B
- `probe_accuracy(will_be_correct) > 0.7` at some (T, L) → strong finding, pursue causally
- `change_point_T*` in first 20% of reasoning → model "knows" early
- `steering_delta_pass@1 > 5pp` → representation is causally relevant
- `activation_patching` restores correct output for >30% → precise causal attribution found

---

## SLURM Infrastructure

All scripts in `interp/scripts/bug_*.sh`. Key scripts:

| Script | Resources | Purpose |
|--------|-----------|---------|
| `bug_build_dataset.sh` | CPU, 30min | Build `pairs.json` from CRUXEval |
| `bug_extract.sh` | 8 GPU fat, 10h | Extract decode-step trajectories |
| `bug_analysis_cpu.sh` | CPU, 6h | Run all Tier 1+2+3 analysis |
| `bug_patch.sh` | 4 GPU fat, 4h | Activation patching (causal) |
| `bug_steer.sh` | 8 GPU fat, 6h | Steering sweep |
| `bug_pipeline.sh` | meta | Chain all jobs with SLURM dependencies |

**Note on fat nodes**: CWM (32B, TP=2) requires 80GB A100s (`-C "fat"`). Standard 40GB nodes have broken IB/RoCE networking on this cluster (observed March 2026) and are not viable.

**Reproducibility**: every script saves `run_config.json` with seed, git hash, date, args, SLURM job ID.

---

## Implementation Status

| Component | Status |
|-----------|--------|
| `interp/bug_trace/mutate.py` | ✅ Done (with SIGALRM timeout) |
| `interp/bug_trace/dataset.py` | ✅ Done |
| `interp/bug_trace/prompts.py` | ✅ Done |
| `interp/bug_trace/run_bug_extract.py` | ✅ Done (position bug fixed) |
| `interp/bug_trace/data/pairs.json` | ✅ Done (830 pairs) |
| `interp/bug_trace/analysis/logit_lens_temporal.py` | ✅ Done |
| `interp/bug_trace/analysis/probe_temporal.py` | ✅ Done |
| `interp/bug_trace/analysis/pca_trajectory.py` | ✅ Done |
| `interp/bug_trace/analysis/change_point.py` | ✅ Done |
| `interp/bug_trace/analysis/ccs.py` | ✅ Done |
| `interp/bug_trace/analysis/activation_patch.py` | ✅ Done |
| `interp/bug_trace/analysis/rsa.py` | ✅ Done |
| `interp/bug_trace/analysis/dmd.py` | ✅ Done |
| `interp/bug_trace/analysis/visualize.py` | ✅ Done |
| `interp/bug_trace/decide_next.py` | ✅ Done |
| `interp/scripts/bug_*.sh` | ✅ Done |
| Extraction job (15901326) | ⏳ PENDING — est. start 23:30 CET Mar 25 |
| Analysis job (15901327) | ⏳ Depends on extraction |
| Patching job (15901328) | ⏳ Depends on analysis |

**Key implementation notes:**
- `run_bug_extract.py`: use `full[n_prompt:]` to extract decode activations — `store.positions` accumulates `n_layers` entries per forward pass and cannot be used for per-layer indexing
- `mutate.py`: `signal.SIGALRM` timeout required — `ConditionFlipMutator` can produce infinite loops
- All analysis scripts: use explicit `None` check (`if h is None: h = traj.get(str(layer))`) not `or` — tensor boolean evaluation raises `RuntimeError`

---

## Data Format

### Trajectory file (`interp-bug-trajectories-track_a/trajectories/{sample_id}.pt`)
```python
{
  "sample_id": str,           # e.g. "orig__sample_0" or "bug__sample_0__off_by_one_minus"
  "pair_id": str,             # links original and its buggy variants
  "original_id": str,
  "is_buggy": bool,
  "mutation_type": str,       # "none" | "off_by_one_minus" | "condition_flip" | ...
  "track": str,               # "track_a" | "track_b"
  "code": str,
  "input_str": str,
  "correct_output": str,
  "generated_text": str,
  "correct": bool,            # did CWM produce the correct fix?
  "n_prompt_tokens": int,
  "decode_token_ids": list[int],   # strided (every 5th generated token)
  "stride": int,              # 5
  "trajectory": {             # hidden states, every 5th decode step
    16: tensor[T, 6144],      # fp16
    32: tensor[T, 6144],
    48: tensor[T, 6144],
    63: tensor[T, 6144],
  }
}
```

Memory: ~10MB/sample × 1280 samples ≈ 12GB total.

---

## Hypotheses

- **H1 (Logit lens)**: P(correct_answer_token) increases over decode steps during Track A reasoning, with a clear "commitment layer" L* at ~layer 40–50.
- **H2 (Probe)**: Hidden states at the "The bug is..." reasoning token carry high `will_be_correct` signal. Earlier tokens carry less.
- **H3 (Change point)**: T* clusters around specific textual events (identifying the bug line, writing the fix).
- **H4 (Steering)**: Adding the fix direction at early decode steps reduces reasoning tokens needed (model commits faster).
- **H5 (Patching)**: The critical (T*, L*) for activation patching corresponds to the same T* found by change point detection — converging evidence.

If H1–H5 are confirmed: CWM maintains a low-dimensional "program fate" encoding in its residual stream, which transitions during reasoning, is causally relevant, and can be localized precisely in (T, L) space.
