# Internal Program Representations During Bug Fixing in CWM

> **Status**: Scaffold — fill in each section as experiments complete.
> Last updated: *auto-updated by analysis scripts*

---

## Abstract

*[To be written after all experiments complete.]*

*(1–2 paragraphs summarising: did we find a representation, how does it evolve,
what's the causal story, what does this mean for our understanding of LLMs?)*

---

## 1. Motivation and Research Questions

### 1.1 The Core Question

When CWM generates tokens on a bug-fixing task — reasoning step by step toward
a fix — does the model maintain an evolving *latent representation* of "which
program it has in mind"? More precisely:

- Does the residual stream at the last token position carry decodable information
  about the correct (fixed) program before the model has output it?
- Does this information evolve directionally over the course of generation,
  starting near a "buggy program" region of representation space and ending
  near a "fixed program" region?
- Is there a single, linearly decodable "program state direction" in the residual
  stream, or is the information distributed across many dimensions?
- Can we causally validate this by steering the model to produce the correct fix
  faster or more reliably?

### 1.2 Why This Is Interesting

Understanding whether LLMs maintain implicit program representations during
chain-of-thought reasoning has implications for:

1. **Mechanistic understanding**: Does the model "think in programs" internally,
   even when reasoning in natural language?
2. **Interpretability tools**: If a clean representation exists, we can track
   and potentially steer the model's "belief" about the program in real time.
3. **Training insights**: Models with a richer internal program representation
   may reason more faithfully — finding this could inform what to train for.
4. **Alignment**: If internal representations diverge from generated text
   ("the model knows the bug but reasons incorrectly"), that's a faithfulness issue.

### 1.3 CWM's Unique Position

CWM is both a natural language reasoning model (via `<think>` tags) and an
execution trace generator (`trace_full` mode). This gives us two complementary
experimental handles:

- **Track A** (NL reasoning): observe how representations evolve during free-form
  reasoning about bugs.
- **Track B** (trace generation): observe how representations evolve during
  structured execution simulation — more interpretable because each trace step
  corresponds to a known execution state.

---

## 2. Experimental Setup

### 2.1 Dataset

**Source**: 800 CRUXEval programs (HuggingFace: `cruxeval-org/cruxeval`).

**Mutation types applied**:
| Mutation | Description | Example | Count | % |
|----------|-------------|---------|-------|---|
| `condition_flip` | Negate if/while condition | `if x:` → `if not x:` | 288 | 34.7% |
| `off_by_one_minus` | Integer constants -1 | `range(n)` → `range(n-1)` | 176 | 21.2% |
| `off_by_one_plus` | Integer constants +1 | `range(n)` → `range(n+1)` | 170 | 20.5% |
| `wrong_comparator` | Swap comparison op | `a < b` → `a <= b` | 115 | 13.9% |
| `wrong_operator` | Swap arithmetic op | `a + b` → `a - b` | 81 | 9.8% |
| **Total** | | | **830** | **100%** |

**Validity filter**: only mutations where the output changes (verified by
execution with 5s timeout). Up to 3 mutations per original program.

**Final dataset**: 830 (buggy, original) pairs from 450 unique CRUXEval programs.
Total samples for extraction: **1280** (450 unique originals + 830 buggy variants,
average 1.84 buggy variants per original program).

**Statistics**:
- Total original programs used: 450 (out of 800 CRUXEval; 350 had no valid mutations)
- Total mutation pairs: 830
- Pairs by mutation type: condition_flip 288, off_by_one_minus 176, off_by_one_plus 170, wrong_comparator 115, wrong_operator 81
- All 830 pairs verified to change the program output (validity filter applied during mutation)

**Dataset file**: `interp/bug_trace/data/pairs.json`

**Reproduction command**:
```bash
python -m interp.bug_trace.mutate \
    --output_path interp/bug_trace/data/pairs.json \
    --n_samples 800 --max_mutations_per_sample 3 --seed 42
```

### 2.2 Model and Inference

**Model**: CWM (32B parameters, 64 layers, hidden dim 6144, GQA, window attention)
**Checkpoint**: `./model_weights/cwm`
**Infrastructure**: SLURM cluster, up to 8× A100 80GB. TP=2, DP=4.

**Track A prompt format**:
```
<system>: Expert Python debugger using <think>...</think>
<user>: "The following Python function has a bug. When called as f({input}),
          it returns {wrong_output} but should return {correct_output}.
          Identify the bug and provide the corrected function."
<assistant>: <think>\n [model generates here]
```

**Track B prompt format**: standard `trace_full` prompt from `evals/cruxeval/prompts.py`,
applied to the *buggy* code. Model generates the execution trace of the buggy program.

**Generation**: greedy decoding (temperature=0), no sampling. CUDA graphs disabled.

**Reproduction command** (Track A, 8 GPUs):
```bash
N_GPUS=8 TRACK=track_a SEED=42 bash interp/scripts/bug_extract.sh
```

### 2.3 Activation Capture

**Layers captured**: [16, 32, 48, 63]

**What is captured**: At each decode step T (one new token), the full hidden state
`h[-1, L]` ∈ ℝ^6144 at the last position for each layer L. This is the model's
complete internal state at that moment.

**Subsampling**: every 5th decode step (stride=5) for memory efficiency.

**Format**: per-sample `.pt` file with `trajectory: {layer → tensor[T/5, 6144]}`

**Memory**: ~10MB per sample, ~12GB total for all samples.

---

## 3. Experiments and Results

### 3.1 Logit Lens Over Decode Steps (Tier 1)

**Methodology**:
At each decode step T and layer L, project `h_T[-1, L]` through the model's
unembedding head (after RMSNorm). The resulting distribution tells us "what
token the model would predict next if it had to commit right now."

Track P(top-1 token) over T for each layer. Find T*(L) = first step where
top-1 probability exceeds a threshold.

**Script**: `interp.bug_trace.analysis.logit_lens_temporal`

**Results**: *(fill in after running)*

**Figure**: `figures/` — commitment surface heatmap (not yet generated)

| Layer | Mean T* (original) | Mean T* (buggy) | Notes |
|-------|--------------------|-----------------|-------|
| 16    | ?                  | ?               | |
| 32    | ?                  | ?               | |
| 48    | ?                  | ?               | |
| 63    | ?                  | ?               | |

**Interpretation**: *(fill in)*

**Scope**:
- Logit lens measures next-token prediction confidence, not full-program representation.
- High top-1 probability at early T means the model commits to a specific token early —
  not necessarily to a specific program.
- Cannot tell us the content of the representation, only its confidence.

---

### 3.2 Linear Probing (Tier 1)

**Methodology**:
At each (layer L, relative time bin B), extract all `h_T[-1, L]` from steps T
in bin B. Train a linear classifier to predict:
- `is_buggy`: does this trajectory come from a buggy program?
- `will_be_correct`: will this generation end with the correct answer?

Produce a 2D accuracy heatmap (T × L).

**Script**: `interp.bug_trace.analysis.probe_temporal`

**Results — is_buggy**: *(fill in)*

| Layer ↓ / Time bin → | bin0 | bin1 | bin2 | ... | bin9 |
|----------------------|------|------|------|-----|------|
| 16                   | ?    | ?    | ?    | ... | ?    |
| 32                   | ?    | ?    | ?    | ... | ?    |
| 48                   | ?    | ?    | ?    | ... | ?    |
| 63                   | ?    | ?    | ?    | ... | ?    |

**Best val_acc (is_buggy)**: ? at layer ?, bin ?
**Best val_acc (will_be_correct)**: ? at layer ?, bin ?

**Figure**: `figures/probe_heatmap_is_buggy.png`, `figures/probe_heatmap_will_be_correct.png`

**Interpretation**: *(fill in)*

**Scope**:
- Linear probe accuracy > 50% means the property is *linearly decodable* from
  hidden states. This is necessary but not sufficient for a "representation" —
  it could still be a side effect of e.g., different token distributions.
- High accuracy at early T (bin 0-2) would be a strong finding: the model
  "knows" early.

---

### 3.3 PCA / Functional PCA (Tier 2)

**Methodology**:
Fit PCA on all mean-pooled trajectory representations. Project each full
trajectory onto the top-2 principal components. Check if PC1 separates
buggy/original (point-biserial correlation).

**Script**: `interp.bug_trace.analysis.pca_trajectory`

**Results**:
*(fill in: explained variance ratio, PBC of PC1 for is_buggy and will_be_correct)*

**Figure**: `figures/pca_trajectories_L{layer}.png`

**Interpretation**: *(fill in)*

**Scope**: PCA finds directions of *maximal variance*, not necessarily directions
meaningful for the task. High PBC is evidence but could reflect other structure
(e.g., token length differences).

---

### 3.4 Change Point Detection (Tier 2)

**Methodology**:
For each trajectory, find T* = the step with the largest inter-step distance
(cosine or L2). This identifies the moment of largest qualitative shift.
Correlate T* with the generated text at that position.

**Script**: `interp.bug_trace.analysis.change_point`

**Results**:

| Layer | Method | Mean T*/T (original) | Mean T*/T (buggy) | Notes |
|-------|--------|----------------------|-------------------|-------|
| 32    | cosine | ?                    | ?                 | |
| 32    | l2     | ?                    | ?                 | |

**Figure**: `figures/change_point_hist_cosine.png`

**Key finding**: *(fill in — does T* cluster at specific reasoning steps?)*

**Interpretation**: *(fill in)*

**Scope**: Change point detection finds the largest single shift, which may not
correspond to the "aha moment" if the shift is gradual. Cosine distance is
normalisation-invariant; L2 is magnitude-sensitive.

---

### 3.5 CCS — Contrast-Consistent Search (Tier 2)

**Methodology**:
For each (original, buggy) pair, find a direction `d` such that
`d · h(original) > d · h(buggy)` consistently. Trains `d` to minimise a
consistency loss without using binary labels.

**Script**: `interp.bug_trace.analysis.ccs`

**Results**:

| Layer | CCS loss | Separation acc | n_pairs | Notes |
|-------|----------|----------------|---------|-------|
| 16    | ?        | ?              | ?       | |
| 32    | ?        | ?              | ?       | |
| 48    | ?        | ?              | ?       | |
| 63    | ?        | ?              | ?       | |

**Figure**: `figures/ccs_separation.png`

**Interpretation**: *(fill in)*

**Scope**: CCS finds a direction consistent with the label *without being told
the labels*. Separation accuracy > 60% strongly suggests a genuine representation
direction. This direction is used for downstream steering and patching.

---

### 3.6 Activation Patching (Tier 3)

**Methodology**:
For buggy-program generations, apply the CCS direction as a steering intervention
at layer L at various points in generation. Check if pass@1 improves.

**Script**: `interp.bug_trace.analysis.activation_patch`

**Results**:

| Layer | T_pos | pass@1 (steered) | pass@1 (unsteered) | Δ |
|-------|-------|------------------|--------------------|---|
| 32    | 0.0   | ?                | ?                  | ? |
| 32    | 0.25  | ?                | ?                  | ? |
| 32    | 0.5   | ?                | ?                  | ? |
| 48    | 0.0   | ?                | ?                  | ? |

**Interpretation**: *(fill in)*

**Scope**: A positive Δ provides causal evidence that the CCS direction is
load-bearing (not just correlated with the outcome). A null result (Δ ≈ 0)
would suggest the representation doesn't causally determine the output, or that
our steering is too coarse-grained.

---

### 3.7 Dynamic Mode Decomposition (Tier 3)

**Methodology**:
Model the trajectory as a linear dynamical system and find dominant temporal modes.
Persistent modes (|λ| ≈ 1) are candidates for "working memory".

**Script**: `interp.bug_trace.analysis.dmd`

**Results**:

| Layer | n_persistent modes (|λ|>0.95) | Top eigenvalue magnitude | Notes |
|-------|-------------------------------|--------------------------|-------|
| 32    | ?                             | ?                        | |
| 48    | ?                             | ?                        | |

**Figure**: `figures/dmd_spectrum.png`

**Interpretation**: *(fill in)*

---

### 3.8 RSA — Representational Similarity Analysis (Tier 3)

**Methodology**:
Compute pairwise cosine similarity between all sample representations and
compare to edit-distance similarity between the programs. High Spearman ρ
means the model's geometry reflects program structure.

**Script**: `interp.bug_trace.analysis.rsa`

**Results**:

| Layer | Spearman ρ(edit dist) | Spearman ρ(same bug status) | N |
|-------|----------------------|------------------------------|---|
| 32    | ?                    | ?                            | ? |
| 48    | ?                    | ?                            | ? |

---

## 4. Synthesis: Is There "One" Representation?

*(To be written after all experiments complete.)*

### What the evidence says

- **Probing**: *(fill in)*
- **CCS**: *(fill in)*
- **Change point**: *(fill in)*
- **Causal validation**: *(fill in)*
- **DMD**: *(fill in)*
- **RSA**: *(fill in)*

### The story

*(1–2 paragraphs: do all methods converge on the same (T*, L*)? Do probing,
CCS and patching agree? Is the representation linear? Is it early or late?)*

### Caveats

- Linear decodability ≠ causal relevance (we tested causality via steering)
- Mean difference / CCS directions may reflect systematic differences in
  *which tokens* appear in buggy vs. original traces, not purely semantic content
- Track A results may reflect NL reasoning artifacts rather than program-level representations
- Sample size (N ≈ 600 pairs) limits statistical power for subtle effects

---

## 5. Open Questions and Future Work

- **Faithfulness**: does the change point T* in representation space coincide
  with where the model *writes* "The bug is..." in text? (Turpin et al. concern)
- **Generality**: do findings generalise beyond CRUXEval-style bugs to real-world
  bugs (SWE-bench)?
- **Causality depth**: activation patching with the CCS direction is coarse.
  Mechanistic circuit analysis (attention head attribution) could give finer causal maps.
- **Sparse autoencoders**: decomposing `h` into sparse features might reveal
  specific "bug type" features (off-by-one, wrong operator, etc.)
- **Track B vs Track A**: do findings differ between trace mode and NL reasoning?
  If so, what does that say about the relationship between world-model and
  language-model reasoning?

---

## Appendix: Reproduction

All results are fully reproducible from:

```bash
# Full pipeline
bash interp/scripts/bug_pipeline.sh

# Or step by step:
bash interp/scripts/bug_build_dataset.sh
bash interp/scripts/bug_extract.sh
bash interp/scripts/bug_analysis_cpu.sh
bash interp/scripts/bug_patch.sh
```

Run configs are saved to `interp-bug-trajectories-{track}/run_config.json`.

Key dependency versions and git commit hash are stored in `run_config.json`.
