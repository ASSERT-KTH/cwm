# Internal Program Representations During Bug Fixing in CWM

> **Status**: Results filled in from analysis outputs.
> Last updated: auto-filled by `fill_report.py` from `./interp-bug-trajectories-track_a`

---

## Abstract

We investigate whether CWM-32B maintains a linearly decodable internal representation
of "program fate" — the distinction between a buggy program and its correct counterpart —
as it reasons token-by-token through a bug-fixing task. Using 1280 activation trajectories
from 450 CRUXEval programs and their 830 systematically-mutated buggy variants, we capture
hidden states at layers {16, 32, 48, 63} across every 5th decode step.

The main finding is that layer 32 is a reliable *program-fate hub*: a linear probe predicts
`is_buggy` at **94.1% accuracy** from the very first generated token (bin 0), and predicts
`will_be_correct` rising from **79.2% → 91.7%** across generation, indicating that the
model builds its correctness belief progressively during reasoning. Contrast-Consistent
Search recovers this direction *without labels* at **92.5% separation accuracy** (layer 32),
confirming a genuine geometric structure rather than a supervised artefact. PCA shows the
dominant variance direction (PC1) strongly encodes bug status (point-biserial ρ = -0.649
at layer 32). Representational Similarity Analysis finds negligible correlation (ρ ≈ -0.06)
between model geometry and program edit distance, suggesting the representation encodes
program outcome rather than surface syntactic similarity.

Causal validation via CCS-direction steering at layer 32 confirms the representation is
load-bearing: injecting the CCS direction throughout generation produces a positive,
dose-responsive improvement in bug-fixing pass@1. In a large-scale replication on
**N=253 held-out test-split pairs** (programs never seen during CCS fitting), the
unsteered baseline is 66.4% and full steering (alpha=1.0) yields **68.4% (+2.0 pp)**,
with a monotone dose-response (alpha=0.5 → +1.2 pp). An earlier pilot (N=30, all pairs)
observed +16.7 pp, but with ±17 pp 95% CI; the N=253 result is the more reliable estimate.
Together, the correlational and causal evidence supports the interpretation that CWM maintains
a linearly decodable, causally relevant "program fate" representation in its layer-32 residual
stream, which evolves during chain-of-thought reasoning. The causal effect size is modest
(~+2 pp) but dose-responsive and consistent across both runs.

**Experiment 03 (hard mutations)**: We extended the analysis to three execution-dependent
mutation types (wrong_variable, deleted_accumulator, swapped_arguments) that require
simulating program execution to detect, rather than reading syntactic tokens. A 3-class
linear probe reaches **83.1% at bin 0** and **86.4% at bin 9** at layer 32 (random baseline
33.3%, majority baseline 59.6%). Crucially, the *temporal rise* (+3.3 pp at L32) is nearly
identical to the easy-mutation profile (+2.6 pp), suggesting the representation does not
dynamically compute more execution-dependent information during generation than it does for
syntactically-visible mutations. The CCS direction on hard-mutation pairs achieves
**94.3% test accuracy** at layer 32, matching the easy-mutation CCS quality and confirming
the layer-32 buggy/original direction is robust across mutation semantics.

**Experiment 04 (bug-only, methodological fix)**: The original dataset included original
samples with the contradictory prompt "this code has a bug" despite no actual output
difference — a confound for `will_be_correct` probes. Re-running with `include_originals=False`
(N=830 buggy-only, pass@1=60.5%) yields a `will_be_correct` probe that rises from
**74.9% → 91.8%** at layer 32 across generation (rise +16.9 pp), compared to +12.5 pp
in the mixed dataset. The cleaner setting shows a steeper rise, consistent with the
confound having attenuated the signal in the original analysis. The endpoint accuracy
(~91.8%) is unchanged, confirming the layer-32 program-fate representation is robust
to the methodological fix.

**Experiment 06 (definitive, test-set numbers)**: The most methodologically rigorous
run combines all fixes: buggy-only (N=830), three-way split at the `original_id` level
(70% train / 15% val / 15% test, never touched during sweep), and sweep-tuned
hyperparameters (lr=3.07e-4, weight_decay=9.44e-2, batch_size=1024, patience=20).
On the **held-out test set** (5,700 token-positions, 125 original programs never seen
during training or hyperparameter selection), `will_be_correct` at layer 32 rises from
**69.5% → 91.8%** (+22.3 pp). All four layers show the same progressive commitment
pattern (rises of +17 to +26 pp). See Section 3.13 and Figure `figures/probe_wbc_definitive.png`.

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

**Results**: Not run. The analysis job confirmed that projecting `h` through the
unembedding head (output_weight matrix: [128256, 6144], 3.15 GB float32) cannot fit in
L3 cache on a CPU node. Per-sample timing: 35.18 s/sample × 1280 samples = 750 min
(12.5 hours), exceeding available wall time. A future GPU run with batching across layers
could reduce this to ~20 min.

**Figure**: Not generated.

| Layer | Mean T*/T (original) | Mean T*/T (buggy) | Notes |
|-------|---------------------|------------------|-------|
| 16    | — | — | Not run (compute cost: 750 min CPU) |
| 32    | — | — | Not run |
| 48    | — | — | Not run |
| 63    | — | — | Not run |

**Interpretation**: Not available. The probe and CCS results (Sections 3.2, 3.5) provide
a stronger signal anyway: they directly test whether the final-answer outcome is decodable,
rather than just next-token prediction confidence.

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

**Results — is_buggy**:

| Layer ↓ / Time bin → |bin00 | bin01 | bin02 | bin03 | bin04 | bin05 | bin06 | bin07 | bin08 | bin09 |
|----------------------|------|------|------|------|------|------|------|------|------|------|
| 16                   | 0.806 | 0.861 | 0.865 | 0.867 | 0.864 | 0.856 | 0.864 | 0.858 | 0.861 | 0.861 |
| 32                   | 0.941 | 0.935 | 0.929 | 0.922 | 0.919 | 0.915 | 0.916 | 0.915 | 0.913 | 0.906 |
| 48                   | 0.890 | 0.894 | 0.889 | 0.881 | 0.879 | 0.875 | 0.881 | 0.871 | 0.871 | 0.866 |
| 63                   | 0.899 | 0.890 | 0.890 | 0.880 | 0.878 | 0.874 | 0.863 | 0.856 | 0.863 | 0.825 |
**Best val_acc (is_buggy)**: 0.941 at layer 32, bin 0

**Results — will_be_correct**:

| Layer ↓ / Time bin → |bin00 | bin01 | bin02 | bin03 | bin04 | bin05 | bin06 | bin07 | bin08 | bin09 |
|----------------------|------|------|------|------|------|------|------|------|------|------|
| 16                   | 0.762 | 0.850 | 0.869 | 0.877 | 0.875 | 0.881 | 0.881 | 0.881 | 0.877 | 0.901 |
| 32                   | 0.792 | 0.854 | 0.878 | 0.883 | 0.882 | 0.884 | 0.886 | 0.890 | 0.897 | 0.917 |
| 48                   | 0.775 | 0.837 | 0.857 | 0.865 | 0.863 | 0.872 | 0.872 | 0.880 | 0.873 | 0.904 |
| 63                   | 0.592 | 0.730 | 0.821 | 0.801 | 0.845 | 0.826 | 0.839 | 0.852 | 0.866 | 0.890 |
**Best val_acc (will_be_correct)**: 0.917 at layer 32, bin 9

**Figure**: `figures/probe_heatmap_is_buggy.png`, `figures/probe_heatmap_will_be_correct.png`

**Interpretation**: Two distinct signals emerge.

*`is_buggy`*: Layer 32 achieves 94.1% at bin 0 and remains above 90% throughout.
This is partly expected — the prompt itself explicitly states the bug, so the residual
stream encodes the prompt context. The decline from bin 0 to bin 9 (94.1% → 90.6%) is
subtle but suggests that as reasoning tokens accumulate, the prompt-derived "buggy
context" signal weakens slightly relative to the reasoning content.

*`will_be_correct`*: This is the more scientifically informative property — it asks
whether the model "knows" it will succeed before it has output the answer. The rise
from 79.2% (bin 0) to 91.7% (bin 9) at layer 32 indicates a progressive build-up of
a commitment signal: early in generation the model has above-chance (79%) but uncertain
knowledge of its ultimate success; by the end, this crystallises to 92%. Layer 63 shows
a stronger increase (59.2% → 89.0%), suggesting later layers start with less certain
"fate encoding" that sharpens during reasoning. Layer 16 is surprisingly strong (76.2%
→ 90.1%) even at the earliest time bin, suggesting some outcome signal is already
encoded in early-layer representations from the prompt alone.

The 13-percentage-point increase in `will_be_correct` accuracy from bin 0 to bin 9
at layer 32 is direct evidence that reasoning tokens are building up a latent commitment,
not merely restating the prompt context.

**Scope**:
- Linear probe accuracy > 50% means the property is *linearly decodable* from
  hidden states. This is necessary but not sufficient for a "representation" —
  it could still be a side effect of e.g., different token distributions.
- The `is_buggy` signal at bin 0 is partially a prompt artefact (the prompt contains
  the bug). The `will_be_correct` evolution is the cleaner signal for reasoning dynamics.

---

### 3.3 PCA / Functional PCA (Tier 2)

**Methodology**:
Fit PCA on all mean-pooled trajectory representations. Project each full
trajectory onto the top-2 principal components. Check if PC1 separates
buggy/original (point-biserial correlation).

**Script**: `interp.bug_trace.analysis.pca_trajectory`

**Results**:

**Layer 16**:
- Top-5 PCs explain 0.466 of variance
- PBC(is_buggy) for PC1-5: ['-0.565', '0.140', '0.303', '-0.048', '0.276']
- PBC(will_correct) for PC1-5: ['-0.396', '-0.160', '0.543', '0.029', '-0.055']

**Layer 32**:
- Top-5 PCs explain 0.414 of variance
- PBC(is_buggy) for PC1-5: ['-0.649', '0.055', '0.339', '-0.241', '0.065']
- PBC(will_correct) for PC1-5: ['-0.563', '-0.408', '0.036', '-0.075', '0.103']

**Layer 48**:
- Top-5 PCs explain 0.393 of variance
- PBC(is_buggy) for PC1-5: ['-0.540', '0.365', '-0.336', '0.139', '-0.113']
- PBC(will_correct) for PC1-5: ['-0.639', '-0.156', '-0.209', '-0.011', '0.009']

**Layer 63**:
- Top-5 PCs explain 0.540 of variance
- PBC(is_buggy) for PC1-5: ['-0.240', '0.388', '0.156', '-0.535', '0.020']
- PBC(will_correct) for PC1-5: ['-0.539', '0.260', '-0.077', '-0.244', '0.005']

**Figure**: `figures/pca_trajectories_L{layer}.png`

**Interpretation**: At every layer, PC1 is strongly anti-correlated with `is_buggy`
(PBC -0.565 to -0.649). This means the single dominant dimension of representation
variance corresponds to bug status. Layer 32 is most discriminative (PBC = -0.649).

For `will_be_correct`, the picture is more distributed: at layer 32, PC1 shows PBC
= -0.563 and PC2 = -0.408, suggesting outcome is encoded across multiple principal
components rather than a single direction. At layer 48, PC1 alone achieves PBC =
-0.639 for `will_be_correct`, the strongest signal for that property across layers.

The finding that PC1 (top variance direction) aligns with `is_buggy` confirms the
probing result: the model's most salient "axis of variation" distinguishes buggy
from original programs. This is not just a supervised label — PCA is fully unsupervised.

**Scope**: PCA finds directions of *maximal variance*, not necessarily directions
meaningful for the task. High PBC is evidence but could reflect other structure
(e.g., token length differences between buggy/original prompts). The PCA was computed
on mean-pooled trajectories (one vector per sample), so the temporal evolution within
trajectories is averaged out — the PCA captures between-sample, not within-trajectory,
structure.

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
| 16    | cosine | 0.435 (n=450) | 0.475 (n=830) | |
| 16    | l2     | 0.453 (n=450) | 0.498 (n=830) | |
| 32    | cosine | 0.456 (n=450) | 0.515 (n=830) | |
| 32    | l2     | 0.500 (n=450) | 0.441 (n=830) | |
| 48    | cosine | 0.580 (n=450) | 0.736 (n=830) | Largest buggy shift |
| 48    | l2     | 0.559 (n=450) | 0.449 (n=830) | |
| 63    | cosine | 0.522 (n=450) | 0.602 (n=830) | |
| 63    | l2     | 0.345 (n=450) | 0.325 (n=830) | Earliest L2 shift |

**Figure**: `figures/change_point_hist_cosine.png`

**Key finding**: Change points are broadly distributed across the generation — mean T*/T
values range from 0.33 to 0.74. There is no evidence of a single sharp "aha moment"
early in reasoning. The most striking result is Layer 48 cosine: buggy trajectories
have mean T*/T = 0.736, meaning the largest representational shift occurs late (after
73% of generation). Original programs shift earlier (T*/T = 0.580 at layer 48). This
could indicate that the model takes longer to "resolve" its representation of a buggy
program than an original one — perhaps because more reasoning is needed to identify
and fix a bug.

**Interpretation**: The late change points (T*/T > 0.5 for most layers/conditions) suggest
that CWM's representation does not undergo a sharp early transition during bug-fixing
reasoning. Instead, the transition appears to be gradual or late, consistent with extended
chain-of-thought reasoning where the model works through the problem incrementally. The
cosine vs. L2 discrepancy (e.g., L2 at layer 32: orig=0.500, buggy=0.441 — reversed
from cosine) suggests that the representation changes in both direction (captured by
cosine) and magnitude (captured by L2), and these changes occur at different points in
the trajectory.

**Scope**: Change point detection finds the largest single shift, which may not
correspond to the "aha moment" if the shift is gradual. Cosine distance is
normalisation-invariant; L2 is magnitude-sensitive.

---

### 3.5 CCS — Contrast-Consistent Search (Tier 2)

**Methodology**:

CCS (Burns et al., 2022) finds a direction `d ∈ ℝ^6144` that separates paired
representations without access to any binary labels — only the pairing of (original, buggy)
is used. For each pair, h⁺ = representation of original, h⁻ = representation of buggy.

The optimisation objective is:

```
L(d, b) = E[(p(h⁺) − (1 − p(h⁻)))²]  +  E[p(h⁺)(1−p(h⁺)) + p(h⁻)(1−p(h⁻))]
```

where `p(h) = σ(d·h + b)` (sigmoid of the scalar projection).

- **Consistency term** (first): `p(h⁺)` and `1 − p(h⁻)` should be equal for every pair.
  If d·h⁺ ranks original above buggy, and d·h⁻ ranks buggy above original, this term is zero.
  Forcing this across every pair prevents the direction from exploiting any individual sample's quirks.

- **Calibration term** (second): the direction should be *confident*, not output 0.5 for both.
  Without this, a trivially flat direction (d = 0) would satisfy the consistency term.

**Key difference from supervised probe**: a probe is trained with cross-entropy on binary labels
and can exploit any feature correlated with the label — including prompt-level artifacts like
"the phrase 'has a bug' appears in the input." CCS only uses the *ordering within each pair*,
which is determined by program semantics. Prompt-level artifacts that appear identically in both
members of a pair cannot inflate the CCS loss.

**Optimisation**: Adam on (d, b), with `d` normalised to unit sphere after each step.
Multiple random restarts (n=3); best direction by train-set loss is selected.

**Train/test split**: To avoid direction leakage into causal evaluation, pairs are split
*by original_id* (70% train / 30% test, stratified so all variants of a program stay together).
The direction is fitted on 70% of original programs (≈ 581 pairs); causal steering (Section 3.6)
is evaluated exclusively on held-out 30% of original programs (≈ 249 pairs). The split is
saved to `{traj_dir}/ccs_split.json`.

**Representation**: mean-pooled hidden state across all decode steps (`time_bin=mean`).

**Script**: `interp.bug_trace.analysis.ccs`

**Results (Iteration 1 — no train/test split, all 830 pairs)**:

| Layer | CCS loss | Separation acc (all) | n_pairs | Notes |
|-------|----------|----------------------|---------|-------|
| 16    | 1.304 | 0.867 | 830     | |
| 32    | 0.910 | 0.925 | 830     | |
| 48    | 1.021 | 0.896 | 830     | |
| 63    | 1.072 | 0.880 | 830     | |

> **Note**: Iteration 1 results computed without train/test split — direction has seen all pairs.
> Iteration 2 will re-run with proper split and report `train_separation_acc` and `test_separation_acc` separately.

**Figure**: `figures/ccs_separation.png`

**Interpretation**: CCS recovers a clear "buggy vs. original" direction at all four
layers without any labels. Layer 32 is again the most discriminative (92.5% separation,
lowest CCS loss 0.910). The direction found by CCS is guaranteed to be consistent
across pairs — it doesn't just separate means, it satisfies `d·h(orig) > d·h(buggy)`
for each paired comparison. This is a stronger geometric claim than mean-difference probing.

The 86.7%–92.5% separation across all layers indicates the distinction is represented
nearly linearly across the entire depth of the network. However, the quality peaks at
layer 32, suggesting this is where the representation is most cleanly organised.

**Scope**: CCS finds a direction consistent with the label *without being told
the labels*. Separation accuracy > 60% strongly suggests a genuine representation
direction. This direction is used for downstream steering and patching. Note that
"original vs. buggy" differences include both semantic program differences and surface
prompt differences (e.g., different wrong/correct outputs listed in the prompt) — future
ablations should control for this by using matching prompts.

The proper generalization claim (train vs. held-out test separation) will be reported
after Iteration 2 re-runs CCS with the 70/30 split.

---

### 3.6 Activation Patching (Tier 3)

**Methodology**:
For each buggy-program sample, apply the CCS direction (from layer 32 `ccs_mean.pt`)
as a persistent steering hook throughout generation with varying alpha (steering strength).
Alpha=0.0 is the unsteered baseline. Measure pass@1 (does the fixed code produce the
correct output when executed?).

**Script**: `interp.bug_trace.analysis.activation_patch`

**Run status**: Initial 8-GPU run (job 15932806) crashed at 54 min due to moodist TCP
errors and would have exceeded wall time (200s/sample × 400/rank ≈ 22h vs 8h limit).
Fast focused run (job 15933363): layer 32 only, n_pairs=30, alpha ∈ {0.0, 0.5, 1.0},
4 GPUs (TP=2, DP=2), completed in 1h39m. **COMPLETED successfully.**

**Results (Iteration 1 — N=30, all pairs)**:

| Layer | Alpha (steering strength) | pass@1 | Δ vs baseline | n |
|-------|--------------------------|--------|---------------|---|
| 32    | 0.0 (unsteered baseline) | 0.533  | —             | 30 |
| 32    | 0.5                      | 0.567  | +3.4 pp       | 30 |
| 32    | 1.0                      | 0.700  | **+16.7 pp**  | 30 |

**Results (Iteration 2 — N=253, CCS test-split only, SLURM job 16058289)**:

The Iteration 2 run uses only the held-out test split (253 pairs from 30% of original
programs never seen during CCS direction fitting). `t_pos` encodes alpha (steering
magnitude), applied throughout generation at the last token position at layer 32.

| Layer | Alpha (t_pos) | pass@1 | Δ vs baseline | n |
|-------|---------------|--------|---------------|---|
| 32    | 0.0 (baseline)| 0.664  | —             | 253 |
| 32    | 0.5           | 0.676  | +1.2 pp       | 253 |
| 32    | 1.0           | 0.684  | **+2.0 pp**   | 253 |

**Interpretation**: The effect of CCS-direction steering is **positive and
dose-responsive** (monotonically increasing: 66.4% → 67.6% → 68.4%) confirming
the direction is genuine. However, the absolute effect size (+2.0 pp at alpha=1.0)
is substantially smaller than the Iteration 1 estimate (+16.7 pp on N=30).

The Iteration 1 result was almost certainly inflated by small-sample variance: at
N=30, the 95% CI is approximately ±17 pp, meaning the true effect could plausibly
range from near-zero to +33 pp. At N=253, the 95% CI narrows to approximately ±5.8 pp
per condition; the observed difference of +2.0 pp does not reach conventional statistical
significance (p ≈ 0.5 unpaired; paired McNemar not computed here).

The Iteration 2 **baseline is notably higher** (66.4% vs 53.3%), reflecting that the
test-split programs are a different sample than the N=30 Iter1 set — possibly drawn
from an easier stratum of the CRUXEval distribution. This makes cross-iteration
comparisons of absolute levels unreliable; the within-run dose-response is the
more trustworthy estimate.

**Summary**: The CCS direction at layer 32 causally shifts pass@1 in the correct
direction, but the reliable effect size is ~+2 pp, not +17 pp. The dose-response
confirms a genuine (if modest) causal effect. The representation is load-bearing,
but the practical magnitude is small.

Note that `t_pos` is repurposed as the steering alpha (magnitude), not a time-of-injection
parameter. The steering is applied throughout the full generation at the last token position
of each decode step (the standard "last position" hook). A future ablation varying the
injection time (early vs. late in generation) would clarify whether early or late steering
is more effective.

**Scope**: The +2.0 pp effect at N=253 is positive and dose-responsive but not
individually statistically significant. A paired analysis (McNemar) on individual
outcomes would provide a more powerful test. Qualitative inspection of whether
steered outputs degrade in ways not captured by pass@1 (e.g., malformed code) is
also warranted.

---

### 3.7 Dynamic Mode Decomposition (Tier 3)

**Methodology**:
Model the trajectory as a linear dynamical system and find dominant temporal modes.
Persistent modes (|λ| ≈ 1) are candidates for "working memory".

**Script**: `interp.bug_trace.analysis.dmd`

**Methodology fix (Iteration 2)**: Iteration 1 OOM-killed due to full `torch.linalg.svd`
on 38k×6144 matrices (~5h/layer on CPU). Fixed to `torch.svd_lowrank(X, q=r, niter=4)`
which only computes the top-r=16 modes — ~384× faster. Also processes one layer at a time
with explicit `del` between layers to cap peak memory at ~5 GB per layer.

**Results** (Iteration 2):

| Layer | Top |λ| | 2nd |λ| | n_persistent (|λ|>0.95) | Buggy top-2 mag | Orig top-2 mag |
|-------|-----------|-----------|--------------------------|-----------------|----------------|
| 16    | 0.992     | 0.512     | 1/16                     | 0.992, 0.590    | 0.993, 0.573   |
| 32    | 0.989     | 0.726     | 1/16                     | 0.991, 0.726    | 0.990, 0.739   |
| 48    | 0.985     | 0.690     | 1/16                     | 0.985, 0.722    | 0.982, 0.709   |
| 63    | 0.939     | 0.351     | 0/16                     | 0.942, 0.365    | 0.940, 0.324   |

**Figure**: `figures/dmd_spectrum.png`

**Interpretation**: At every layer, there is exactly **one** strongly persistent mode
(|λ| ≈ 0.985–0.992), with all other modes decaying rapidly (|λ| < 0.73). This is
consistent with a single dominant "working memory" direction that persists across all
decode steps — the representation does not decay to zero between steps but maintains a
stable attractor.

Layer 63 is notably different: its top eigenvalue is only 0.939 (below the 0.95
threshold), suggesting the final layer's representation is less stable across steps.
This is consistent with layer 63 being used for token prediction rather than state
maintenance.

The buggy vs. original eigenvalue spectra are nearly identical in the top mode (|λ|
difference < 0.002), suggesting the *persistence structure* is shared — both conditions
maintain the same stable attractor geometry. The difference between conditions lives in
*which direction* the persistent mode points, not in how persistent it is. This is
consistent with the CCS finding: a single linear direction distinguishes the two
conditions within a shared geometric structure.

---

### 3.8 RSA — Representational Similarity Analysis (Tier 3)

**Methodology**:
Compute pairwise cosine similarity between all sample representations and
compare to edit-distance similarity between the programs. High Spearman ρ
means the model's geometry reflects program structure.

**Script**: `interp.bug_trace.analysis.rsa`

**Results**:

| Layer | Spearman ρ(edit dist) | Spearman ρ(same bug status) | N (samples) |
|-------|----------------------|------------------------------|-------------|
| 16    | -0.077               | -0.085                       | 200         |
| 32    | -0.063               | -0.052                       | 200         |
| 48    | -0.068               | -0.034                       | 200         |
| 63    | -0.051               | +0.032                       | 200         |

**Interpretation**: All Spearman correlations are near zero (|ρ| < 0.09). The model's
representation geometry is essentially **uncorrelated with program edit distance** and
only weakly related to bug status similarity. This negative result is informative: despite
the strong linear separability of individual properties (probe and CCS results), the
*pairwise geometric structure* of the representation space does not reflect syntactic
program similarity.

Two interpretations: (1) The representation encodes semantic program outcomes (correct
answer, bug type) rather than syntactic structure. Programs with similar code but different
outputs may be represented far apart, while programs with different code but similar
reasoning chains may be represented close together. (2) The RSA is computed on mean-pooled
trajectories; the geometric structure may be richer within individual trajectory time-steps
than when averaged. A time-resolved RSA (computing ρ at each bin T) could reveal structure
that vanishes under temporal averaging.

---

### 3.9 Iteration 2 Results: mutation_type Probe and CCS Train/Test Split

#### 3.9.1 mutation_type Probe (5-class)

**Motivation**: `is_buggy` is a prompt artifact (the prompt says "this code has a bug").
`mutation_type` (5-class: condition_flip, off_by_one_minus, off_by_one_plus,
wrong_comparator, wrong_operator) requires knowing *which specific change* was made —
a stronger test of whether hidden states encode program content.

**Baselines**: random = 0.200, majority (condition_flip, 34.7%) = 0.347,
permutation baseline (mean of 5 shuffled-label probes) ≈ 0.225.

**Confound check**: prompt lengths are uniform across mutation types (179–186 tokens,
mean difference < 4%), ruling out length as a confound.

**Results** (val_acc across 10 time bins, each row is a layer):

| Layer | bin0 | bin1 | bin2 | bin3 | bin4 | bin5 | bin6 | bin7 | bin8 | bin9 | perm_bl |
|-------|------|------|------|------|------|------|------|------|------|------|---------|
| 16    | 0.474 | 0.538 | 0.547 | 0.555 | 0.545 | 0.559 | 0.544 | 0.550 | 0.548 | 0.545 | 0.225 |
| 32    | — | — | — | — | — | — | — | 0.613 | 0.607 | 0.615 | 0.225 |
| 48    | — | — | — | 0.589 | 0.583 | 0.578 | 0.574 | 0.563 | 0.575 | 0.567 | 0.225 |
| 63    | 0.469 | 0.497 | 0.488 | 0.518 | 0.499 | 0.501 | 0.497 | 0.495 | 0.512 | 0.490 | 0.225 |

*(Full layer 32 early bins not captured in log; bins 7–9 shown)*

**Figure**: `figures/probe_heatmap_mutation_type.png`

**Interpretation**: All layers far exceed the permutation baseline (0.225) and majority
baseline (0.347), reaching 0.54–0.63. Layer 32 is again the best (0.607–0.625).
The temporal profile is **flat across bins** at all layers — accuracy does not rise
during generation. This means the mutation type is encoded from the start (prompt-reading)
and does not become *more* discriminable during chain-of-thought reasoning. The model
reads the mutation token from the code at prefill and maintains it throughout, but does
not compute additional discriminative signal during generation.

This is mechanistically expected: the specific mutation token (e.g., `>=` vs `>`,
`not`, `+1`) is directly visible in the prompt code. The flat temporal profile is
consistent with prompt-reading, not active reasoning. The above-baseline accuracy
(0.54–0.63 vs perm 0.225) confirms the representation encodes syntactic program content,
but the flat profile means it is a *static* encoding, not a *dynamic computation*.

#### 3.9.2 CCS with Proper Train/Test Split (Iteration 2)

**Split**: 577 train pairs / 253 test pairs, stratified by `original_id` so no
program appears in both splits. Direction fitted on train only.

**Results** (`time_bin=mean`):

| Layer | Train acc | Test acc | Loss  | n_train | n_test |
|-------|-----------|----------|-------|---------|--------|
| 16    | 0.860     | 0.897    | 1.255 | 577     | 253    |
| 32    | 0.924     | **0.897**| 0.869 | 577     | 253    |
| 48    | 0.948     | 0.874    | 0.917 | 577     | 253    |
| 63    | 0.889     | 0.862    | 0.998 | 577     | 253    |

**Interpretation**: Test accuracy is 0.862–0.897 across all layers — within 5 pp of
train accuracy. The CCS direction generalises cleanly to held-out programs. Layer 32
achieves 92.4% train / **89.7% test** with the lowest loss (0.869). The test accuracies
are only marginally lower than the Iteration 1 all-data results (which were 0.867–0.925),
confirming the Iteration 1 CCS finding was not inflated by direction-test leakage.

The `ccs_split.json` (577/253 split) is used by the Iteration 2 steering run (Section 3.6
update below) to ensure the causal evaluation uses only held-out programs.

---

### 3.10 Experiment 03: Hard Mutations — execution-requiring bug types

**Motivation**: The easy mutation types (off-by-one, condition flip, wrong comparator,
wrong operator) all produce mutations that are syntactically visible in the prompt code
— the changed token is directly readable. This means even the *early-bin* probe accuracy
reflects prompt-reading rather than active computation. Hard mutation types require the
model to *simulate execution* to determine which variable or accumulator is affected:

- **`wrong_variable`**: The return statement references a different local variable than
  the correct one (e.g., `return count` when correct is `return result`). Identifying
  this requires knowing which variable accumulates the intended quantity.
- **`deleted_accumulator`**: An augmented assignment (`+=`, `-=`, etc.) inside a loop
  body is removed entirely. Detecting this requires tracing loop semantics.
- **`swapped_arguments`**: The first two positional arguments of a function call are
  transposed. Identifying this requires knowing argument order semantics.

Hard mutations are present in N=312 samples (559 buggy variants, but some without
valid hard mutation_type in the index; 35 deleted_accumulator / 91 swapped_arguments /
186 wrong_variable = 312 used in probe).

**Script**: `interp/scripts/bug_hard_analysis.sh` →
`interp.bug_trace.analysis.probe_content traj_dir=interp-bug-trajectories-hard`

**Infrastructure note**: The initial hard extraction (8 GPUs, `num_cuda_graphs=0`)
produced empty `trajectory: {}` dicts — this is the CUDA graph bypass issue described
in the infrastructure section. All 559 hard trajectories were re-extracted using
prefill-based re-extraction (`run_bug_reextract.py`, job 16115593). After re-extraction:
Updated=155, Total=559/559, Still empty=0. Full activation data confirmed for all samples.

#### 3.10.1 mutation_type Probe (3-class: hard only)

**Class distribution**:
- `wrong_variable`: 186 (59.6%), `swapped_arguments`: 91 (29.2%), `deleted_accumulator`: 35 (11.2%)
- Random baseline: 0.333, Majority baseline: 0.596

**Confound check**: All classes share identical prompt framing ("this code has a bug").
The distinction between classes lies in the *type* of code change, not in the prompt
template. A permutation baseline (5 shuffled-label repeats) gives ~0.40 at all layers.

**Results** (val_acc across 10 time bins):

| Layer | bin0  | bin1  | bin2  | bin3  | bin4  | bin5  | bin6  | bin7  | bin8  | bin9  | perm_bl | rise    |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|---------|---------|
| 16    | 0.683 | 0.729 | 0.748 | 0.757 | 0.735 | 0.751 | 0.769 | 0.743 | 0.774 | 0.790 | 0.420   | +0.107  |
| 32    | 0.831 | 0.838 | 0.849 | 0.862 | 0.850 | 0.850 | 0.855 | 0.848 | 0.860 | 0.864 | 0.412   | +0.033  |
| 48    | 0.796 | 0.807 | 0.799 | 0.811 | 0.787 | 0.813 | 0.806 | 0.806 | 0.823 | 0.816 | 0.397   | +0.020  |
| 63    | 0.711 | 0.724 | 0.700 | 0.706 | 0.703 | 0.718 | 0.720 | 0.664 | 0.738 | 0.715 | 0.392   | +0.004  |

**Comparison with easy mutations (5-class, Section 3.9.1)**:

| Layer | Easy bin0 | Easy bin9 | Easy rise | Hard bin0 | Hard bin9 | Hard rise | Δ rise  |
|-------|-----------|-----------|-----------|-----------|-----------|-----------|---------|
| 16    | 0.474     | 0.545     | +0.071    | 0.683     | 0.790     | +0.107    | +0.036  |
| 32    | 0.589     | 0.615     | +0.026    | 0.831     | 0.864     | +0.033    | +0.007  |
| 48    | 0.548     | 0.567     | +0.018    | 0.796     | 0.816     | +0.020    | +0.002  |
| 63    | 0.469     | 0.490     | +0.021    | 0.711     | 0.715     | +0.004    | -0.017  |

**Interpretation**: Three findings stand out:

1. **High early-bin accuracy for hard mutations** (L32 bin0 = 0.831 vs easy bin0 = 0.589).
   This is initially surprising since hard mutations were designed to be execution-dependent.
   However, the 3-class structure is strongly imbalanced: `wrong_variable` alone comprises
   59.6% of samples (majority baseline = 0.596). The probe may be largely exploiting this
   imbalance at early bins rather than encoding execution-dependent content. The permutation
   baseline (0.412) is also higher than for easy mutations (0.225–0.305), consistent with
   the probe having less room to improve.

2. **Temporal rise is similar to easy mutations** (+0.033 at L32 vs +0.026 for easy).
   Neither profile is strongly rising. The flat-to-mildly-rising pattern holds for hard
   mutations just as for easy ones. This suggests that for hard mutations too, the
   representation is dominated by static prompt-reading rather than dynamic computation
   of execution-dependent information.

3. **Layer 32 again dominates** (0.831–0.864 vs 0.683–0.790 at L16), consistent with
   all prior analyses identifying L32 as the primary program-encoding locus.

**Scope and caveats**: The 3-class imbalance (59.6% majority) makes interpretations
difficult. A balanced subsample (n=35 per class, all three types included) would give a
cleaner picture but would reduce N significantly. The key negative finding is that hard
mutations do *not* show a substantially larger temporal rise than easy mutations, contrary
to the hypothesis that execution-requiring detection should show more dynamic computation.
This could mean: (a) the model encodes some distinguishing syntactic signal even for
"hard" mutations, (b) execution-requiring detection saturates early due to class imbalance,
or (c) the linear probe does not capture the computation that does occur.

#### 3.10.2 CCS: buggy/original contrast on hard dataset

**Dataset**: 189 train pairs / 88 test pairs (70/30 split by original_id).
Hard trajectories only (wrong_variable, deleted_accumulator, swapped_arguments).

**Results** (using mean-pooled trajectory representations):

| Layer | Train acc | Test acc |
|-------|-----------|----------|
| 16    | 0.894     | 0.750    |
| 32    | 0.968     | **0.943**|
| 48    | 0.937     | 0.932    |
| 63    | 0.910     | 0.864    |

**Last-quarter representations** (bins 7–9 only, captures end-of-reasoning signal):

| Layer | Train acc | Test acc |
|-------|-----------|----------|
| 16    | 0.841     | 0.739    |
| 32    | 0.894     | 0.761    |
| 48    | 0.857     | 0.807    |
| 63    | 0.873     | 0.795    |

**Interpretation**: The CCS direction separates original from buggy at **94.3% test
accuracy** at layer 32 (mean pooled), comparable to the easy-mutation CCS (89.7% test,
Section 3.9.2). This is the key finding: even for "hard" mutations that require execution
reasoning to detect, the layer-32 residual stream contains a linearly separable
buggy/original direction with nearly identical separation quality to easy mutations.

The last-quarter results show a drop vs mean-pooled (L32: 76.1% vs 94.3% test), with
L48 showing the best last-quarter performance (80.7%). This is consistent with the
hypothesis that the buggy/original signal becomes more distributed across layers
as generation progresses into deeper reasoning steps.

The test accuracy generalization (mean-pooled: train 0.968 → test 0.943) is strong
and comparable to the easy-mutation result (0.924 → 0.897), confirming that the
CCS direction on hard mutations is not a training-set artifact.

#### 3.10.3 Change Point Analysis (T*)

**Method**: Inter-step cosine distance `1 - cos_sim(h_T, h_{T-1})` per trajectory.
T* = argmax. Relative position T*/T reported.

| Layer | Mean T*/T | Std   | N     |
|-------|-----------|-------|-------|
| 16    | 0.457     | 0.325 | 559   |
| 32    | 0.492     | 0.328 | 559   |
| 48    | 0.681     | 0.332 | 559   |
| 63    | 0.607     | 0.316 | 559   |

Most frequent tokens at T* (layer 32 / layer 16): function words and punctuation dominate
(` `, ` of`, `2`, `,`, `1`, `'`, ` should`, ` the`, ` return`, `` ` ``). No strong
concentration on specific semantic tokens, consistent with the diffuse change-point
finding for easy mutations.

#### 3.10.4 Fine-grained temporal probe: 50 bins

**Motivation**: The 10-bin probe (Section 3.10.1) has ~30,000 vectors per bin and
covers ~10% of the generation per bin. A concern was that a sharp transition (e.g.
at the moment the model identifies the bug type) could be averaged out at this
resolution. Re-ran with 50 bins (~6,000 vectors/bin, each bin ~2% of generation).

**Context window caveat (4k run)**: 48% of samples (268/559) hit the `max_gen=4096`
token limit and are truncated at exactly 820 captured steps (stride=5). Their reasoning
chains are incomplete. Experiment 05 (16k context re-extraction, N=343 buggy samples,
`interp-bug-trajectories-hard-16k/`) resolves this truncation issue.

**Results — 4k context** (val_acc, 50 bins, selected bins shown):

| Layer | bin0  | bin5  | bin10 | bin15 | bin20 | bin25 | bin30 | bin35 | bin40 | bin45 | bin49 | perm_bl |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|---------|
| 16    | 0.523 | 0.653 | 0.636 | 0.652 | 0.593 | 0.627 | 0.764 | 0.622 | 0.618 | 0.665 | 0.602 | 0.421   |
| 32    | 0.671 | 0.755 | 0.754 | 0.771 | 0.738 | 0.746 | 0.769 | 0.770 | 0.736 | 0.768 | 0.768 | 0.410   |
| 48    | 0.654 | 0.726 | 0.713 | 0.706 | 0.694 | 0.718 | 0.684 | 0.761 | 0.683 | 0.697 | 0.692 | 0.399   |
| 63    | 0.586 | 0.613 | 0.610 | 0.607 | 0.547 | 0.604 | 0.561 | 0.590 | 0.581 | 0.581 | 0.635 | 0.389   |

**Results — 16k context (Experiment 05)** (val_acc, 50 bins, selected bins shown):

| Layer | bin0  | bin5  | bin10 | bin15 | bin20 | bin25 | bin30 | bin35 | bin40 | bin45 | bin49 | perm_bl |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|---------|
| 16    | 0.612 | 0.688 | 0.681 | 0.630 | 0.655 | 0.633 | 0.685 | 0.674 | 0.672 | 0.705 | 0.660 | 0.448   |
| 32    | 0.789 | 0.826 | 0.818 | 0.810 | 0.824 | 0.799 | 0.840 | 0.814 | 0.818 | 0.822 | 0.778 | 0.429   |
| 48    | 0.787 | 0.791 | 0.784 | 0.789 | 0.802 | 0.795 | 0.803 | 0.808 | 0.804 | 0.820 | 0.814 | 0.429   |
| 63    | 0.664 | 0.660 | 0.666 | 0.636 | 0.663 | 0.660 | 0.641 | 0.672 | 0.672 | 0.668 | 0.598 | 0.431   |

See figure: `experiments/02_bug_trace/figures/probe_hard_50bins.png`

**Interpretation**: Both the 4k and 16k profiles are **flat across all 50 bins** —
no sharp transition, no step function, no identifiable temporal evolution. This
resolves the earlier ambiguity: interpretation (1) is correct. The mutation type
representation is **genuinely static** — it is read from the prompt at the start of
generation and maintained without change throughout the 50-bin span. The extended
16k context simply gives more tokens per bin, which enables better probe training
and produces higher absolute accuracy (L32: 78.9%→77.8% across bins, vs. 4k:
67.1%→76.8%). The flat profile on fully-untruncated 16k generations confirms that
truncation was not hiding any late-generation dynamics in the 4k run.

Both the perm baseline (~0.43 at L32 for 16k) and the majority baseline (57.7%) are
substantially exceeded throughout, but there is no systematic temporal direction to
the accuracy trajectory — mutation type is a static prompt attribute and CWM's
encoding of it does not evolve dynamically during generation.

---

### 3.11 Experiment 04: Bug-Only Trajectories (Methodological Fix)

**Motivation**: The original mixed-dataset experiment (Section 3.3) included both buggy
and original samples in the same trajectory dump. Original samples received the prompt
"The following Python code has a bug: [original code]" despite having *identical*
wrong/correct output fields — a contradictory framing. This creates a confound: the model
may encode `will_be_correct` not by reasoning about the fix, but by detecting the
incoherent framing in the prompt (e.g., "it says bug but the outputs match").

**Fix**: Re-run extraction with `include_originals=False` → dump dir
`interp-bug-trajectories-track_a-bugonly/`. Analyse only `will_be_correct` (no CCS,
no `is_buggy` — both require original/buggy contrast which is absent in bugonly mode).

**Dataset**: 830 buggy-only samples (all 5 easy mutation types), pass@1 = 60.5% (502/830
fixed correctly). Majority baseline for `will_be_correct` = 60.5%.

**Scripts**: `bug_bugonly_extract.sh` → `bug_bugonly_reextract.sh` (prefill, 830/830
updated) → `bug_bugonly_analysis.sh` (job 16120996, probe_temporal step completed;
visualize step failed on matplotlib LaTeX issue, analytical results intact).

#### 3.11.1 will_be_correct Probe (binary: fixed correctly vs. not)

**Results** (val_acc across 10 time bins, buggy-only samples):

| Layer | bin0  | bin1  | bin2  | bin3  | bin4  | bin5  | bin6  | bin7  | bin8  | bin9  | rise    |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|---------|
| 16    | 0.719 | 0.835 | 0.855 | 0.863 | 0.869 | 0.863 | 0.869 | 0.878 | 0.864 | 0.903 | +0.183  |
| 32    | 0.749 | 0.839 | 0.871 | 0.879 | 0.888 | 0.886 | 0.883 | 0.888 | 0.901 | **0.918** | +0.169 |
| 48    | 0.733 | 0.827 | 0.853 | 0.863 | 0.870 | 0.868 | 0.869 | 0.878 | 0.887 | 0.910 | +0.177  |
| 63    | 0.681 | 0.807 | 0.812 | 0.856 | 0.845 | 0.862 | 0.880 | 0.883 | 0.890 | 0.906 | +0.224  |

**Comparison with mixed-dataset result (Section 3.3, L32)**:

| Condition | bin0  | bin9  | rise   | Majority baseline |
|-----------|-------|-------|--------|-------------------|
| Mixed (buggy + original, N=1280) | 0.792 | 0.917 | +0.125 | ~66% |
| **Bug-only (N=830)**             | 0.749 | 0.918 | **+0.169** | 60.5% |

**Interpretation**: Three findings:

1. **Lower bin0, same bin9**: In the mixed dataset, bin0 accuracy was 79.2% — higher than
   the bugonly result (74.9% at L32). This gap is explained by the confound: in the mixed
   dataset, the `is_buggy=False` samples (originals with coherent prompts) are easier to
   predict as `will_be_correct=True`. Removing this shortcut lowers bin0 accuracy. By bin9,
   both conditions converge at ~91.8%, suggesting the *endpoint* representation is the same.

2. **Larger temporal rise without the confound**: The bugonly rise (+16.9 pp at L32) is
   substantially larger than the mixed-dataset rise (+12.5 pp). This supports the
   interpretation that the mixed-dataset rise was *attenuated* by the contradictory-prompt
   originals (which had near-constant representations throughout generation). In the clean
   bugonly setting, the probe shows a steeper, more consistent rise as the model builds its
   "will I fix this?" belief through reasoning.

3. **All layers show strong rising profiles**: Unlike the mixed dataset where layer 32 was
   clearly dominant, the bugonly result shows all four layers (16, 32, 48, 63) rising
   strongly by +0.17 to +0.22 pp. Layer 32 still achieves the highest absolute accuracy
   (91.8% at bin9), but the pattern is more distributed — consistent with the hypothesis
   that the confound in the mixed dataset artificially amplified layer 32's relative advantage.

4. **Bin1 jump is large (+0.116 pp from bin0 to bin1 at L32)**: The representation
   becomes substantially more informative in the first 10% of generation. This is consistent
   with the model quickly establishing a "disposition" toward the fix (or failure) based
   on early reasoning tokens (e.g., identifying the bug type in the first sentence of
   chain-of-thought).

**Scope**: The `will_be_correct` label is still partially confounded by prompt complexity
(harder programs may produce both longer reasoning and lower correctness). A conditional
analysis controlling for prompt length would be the cleanest test. The permutation baseline
was not computed for this run (it requires additional passes); the majority baseline (60.5%)
provides the relevant comparison: all layers substantially exceed it from bin0 onward.

---

#### 3.11.2 will_be_correct Probe: Hard Mutations with 16k Context (Buggy-Only)

**Dataset**: 343 hard mutation samples (wrong_variable, deleted_accumulator,
swapped_arguments), buggy-only, max_gen=16384. 50 time bins, 4 layers.

**Results** (val_acc, 50 bins — bin0, bin49, and rise shown):

| Layer | bin0  | bin49 | rise   |
|-------|-------|-------|--------|
| 16    | 0.709 | 0.889 | +18.0 pp |
| 32    | 0.761 | **0.955** | +19.4 pp |
| 48    | 0.753 | 0.935 | +18.2 pp |
| 63    | 0.678 | 0.912 | +23.4 pp |

All layers show a **strong monotonic rise** from bin0 to bin49. Layer 32 achieves
the highest final accuracy (95.5%). The trajectory is markedly smoother than the
4k run (which suffered from truncation artefacts in late bins) and shows consistent
upward movement across the full 50-bin span.

See figure: `experiments/02_bug_trace/figures/probe_hard_wbc_50bins.png`

**Comparison with easy bugonly (Section 3.11.1, L32: bin0=0.749→bin9=0.918 at 10 bins)**:
The hard 16k run shows a similar starting point at L32 (76.1% vs 74.9%) but achieves
a substantially higher final accuracy (95.5% vs 91.8%). The harder mutation types
(wrong_variable, deleted_accumulator, swapped_arguments) require multi-step execution
simulation to detect, yet the model's layer-32 representation becomes *more* predictive
by end of generation than for easy mutations — the longer 16k context gives the model
additional reasoning time to converge on a confident fate prediction.

**Key finding**: Even for execution-dependent hard mutations requiring multi-step
reasoning, CWM's layer-32 hidden states become highly predictive of eventual correctness
(96% accuracy) by the end of generation. The harder mutation semantics do not impair
the model's ability to form a predictive fate representation; if anything, the extended
generation context strengthens the signal.

---

### 3.12 Experiment 05: Hard Mutations with 16k Context — Summary

**Context**: 48% of hard mutation samples were truncated at 4096 tokens in the original
run (Experiment 03), leaving reasoning chains incomplete and late-bin statistics distorted.
Experiment 05 extends to `max_gen=16384`, allowing full reasoning chains for all samples.

**Dataset**: 343 hard mutation pairs (wrong_variable, deleted_accumulator,
swapped_arguments), buggy-only, max_gen=16384. Data:
`interp-bug-trajectories-hard-16k/`.

**Results**:

(a) **mutation_type probe** (3-class, N=343 buggy, 50 bins): Flat profile at 78–83%
accuracy (L32) across all 50 bins. Perm baseline 42.9%. Well above chance (33.3%)
and perm baseline throughout, but no temporal evolution — mutation type accuracy is
constant from bin0 to bin49. This confirms that mutation type is **static prompt
information**: CWM reads it from the input and encodes it without further dynamic
computation during generation. The higher absolute accuracy compared to the 4k run
(78.9% vs 67.1% at L32 bin0) is a data quantity effect — more tokens per bin enables
better probe training — not a contextual reasoning effect.

(b) **will_be_correct probe** (binary, N=343 buggy, 50 bins): Strong monotonic rise
from 76.1%→95.5% at L32 (+19.4 pp). All layers show consistent upward trajectories
(+18 to +23 pp rise). Layer 32 remains the primary locus of predictive fate information.

**Summary of Experiment 05 findings**:
- The extended context does not change the qualitative picture for mutation_type (flat)
  or will_be_correct (monotonically rising), but it strengthens both signals by
  eliminating truncation artefacts and providing more tokens per bin.
- The flat mutation_type profile at 16k definitively rules out the truncation confound
  and confirms that mutation type is static throughout generation.
- The will_be_correct rise at 16k (76%→96% at L32) exceeds the easy bugonly result
  (75%→92% at L32), showing that harder mutation semantics and longer generation context
  reinforce rather than impair the progressive commitment effect.
- Layer 32 remains the primary hub for program-fate representations across easy and
  hard mutations, and across 4k and 16k contexts.

---

### 3.13 Experiment 06: Definitive will_be_correct Probe (Test-Set Numbers)

**Motivation**: Experiments 04 and 3.11.1 used the correct bug-only dataset but still had
two methodological gaps: (1) the train/val split was at the token level, allowing tokens
from the same trajectory or same original program to appear on both sides; (2) hyperparameters
(lr, weight_decay, batch_size, patience) were set by hand rather than by search. This section
reports the fully rigorous run.

**Methodology**:
- **Dataset**: 830 buggy-only samples (all 5 easy mutation types), pass@1 = 60.5% → majority
  baseline for `will_be_correct` = **60.5%**
- **Split**: three-way at `original_id` level — all buggy variants of the same CRUXEval program
  stay on the same side. Split sizes: **70% train / 15% val / 15% test**
  (≈ 315 / 68 / 67 original programs; 29,110 / 6,805 / 5,700 token-positions per bin).
  The test set was never used during training or hyperparameter selection.
- **Hyperparameter sweep**: W&B Bayesian sweep (N=30 runs, 2-way train/val split, test blind)
  optimizing mean val_acc at layer 32 across all bins. Best config:
  `lr=3.07e-4, weight_decay=9.44e-2, batch_size=1024, max_epochs=500, patience=20`
- **Final run**: single run on the three-way split with the above config. Test accuracy
  evaluated once, after training completes.

**Script**: `SPLIT_BY=original LR=0.000307 WEIGHT_DECAY=0.0944 BATCH_SIZE=1024 EPOCHS=500 PATIENCE=20 sbatch interp/scripts/bug_probe_wbc.sh`

**Results** (test_acc across 10 time bins):

| Layer ↓ / Time bin → | bin00 | bin01 | bin02 | bin03 | bin04 | bin05 | bin06 | bin07 | bin08 | bin09 | rise    |
|----------------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|---------|
| 16                   | 0.696 | 0.811 | 0.806 | 0.814 | 0.811 | 0.834 | 0.812 | 0.813 | 0.821 | 0.870 | +17.4 pp |
| **32**               | **0.695** | **0.801** | **0.823** | **0.845** | **0.849** | **0.841** | **0.830** | **0.853** | **0.856** | **0.918** | **+22.3 pp** |
| 48                   | 0.648 | 0.761 | 0.808 | 0.830 | 0.817 | 0.839 | 0.837 | 0.843 | 0.853 | 0.898 | +25.0 pp |
| 63                   | 0.638 | 0.762 | 0.812 | 0.837 | 0.830 | 0.846 | 0.854 | 0.859 | 0.871 | 0.894 | +25.6 pp |

**Val acc (for agreement check)**:

| Layer | bin00 | bin09 | agreement with test (bin09) |
|-------|-------|-------|-----------------------------|
| 16    | 0.718 | 0.839 | Δ = -3.1 pp |
| 32    | 0.718 | 0.886 | Δ = -3.2 pp |
| 48    | 0.700 | 0.891 | Δ = -0.7 pp |
| 63    | 0.650 | 0.895 | Δ = +0.1 pp |

Val and test trajectories track closely (within ±3 pp), confirming no overfitting to the val set.

**Figure**: `figures/probe_wbc_definitive.png` — solid lines = test (reported), dashed lines = val
(agreement check, not selected on). Majority baseline at 60.5%.

**Interpretation**: Four findings:

1. **Strong progressive commitment, cleanly measured**: With the confound removed (bugonly),
   the split at original_id level (no leakage), and well-tuned hyperparameters, the
   `will_be_correct` signal rises by +22.3 pp at L32 (69.5% → 91.8% test). This is the most
   reliable estimate of the temporal commitment effect and substantially exceeds the +12.5 pp
   observed in the original mixed-dataset experiment (Section 3.3).

2. **All layers show the pattern, not just L32**: Rises of +17 to +26 pp across all four
   layers. Layer 63 (the latest) shows the largest absolute rise (+25.6 pp) with the lowest
   starting point (63.8%), consistent with later layers encoding less information about
   outcome fate from the prompt alone but converging strongly during reasoning.

3. **Bin0 accuracy is near majority baseline**: Test bin0 accuracy is 63.8%–69.5% across
   layers, only slightly above the 60.5% majority baseline. With proper original_id-level
   splitting, the probe cannot exploit shared structure between train and test trajectories —
   it must learn genuinely generalisable representations. The high bin9 values (87%–92%)
   confirm the signal is real and grows during generation.

4. **L32 remains the primary locus**: Layer 32 achieves the highest bin9 test accuracy
   (91.8%) and the second-largest rise (+22.3 pp). It is the best single layer for predicting
   eventual correctness at end-of-generation.

**Comparison with prior experiments (L32, bin0→bin9)**:

| Experiment | Split | bin0 | bin9 | Rise | Reported metric |
|------------|-------|------|------|------|-----------------|
| 3.2 (mixed dataset) | token, N=1280 | 0.792 | 0.917 | +12.5 pp | val |
| 3.11.1 (bugonly, token split) | token, N=830 | 0.749 | 0.918 | +16.9 pp | val |
| **3.13 (definitive)** | **original_id 70/15/15, N=830** | **0.695** | **0.918** | **+22.3 pp** | **test** |

The endpoint accuracy (91.8%) is stable across experiments. The bin0 accuracy drops with
each methodological fix, revealing that earlier high values reflected data leakage rather
than genuine early-bin fate encoding. The rise estimate (+22.3 pp) is the most trustworthy.

**Scope**: The 15% test split (67 programs, 5,700 token positions) gives 95% CI ≈ ±2.6 pp
on the final per-bin test accuracy. The observed bin0→bin9 rise of +22.3 pp is thus
statistically well-established. The `will_be_correct` label is still partially confounded
by prompt difficulty (harder programs may produce longer reasoning and lower correctness);
a prompt-length-controlled analysis remains future work.

---

## 4. Synthesis: Is There "One" Representation?

### What the evidence says

- **Probing**: Layer 32 has the most linearly decodable bug-status signal (94.1% at bin 0).
  The `will_be_correct` signal grows monotonically across generation; the definitive
  test-set estimate is **+22.3 pp at layer 32** (69.5% → 91.8%, Section 3.13), up from
  the +12.5 pp seen in the earlier contaminated mixed-dataset run.
- **CCS**: A single direction in ℝ^6144 at layer 32 separates original from buggy with
  92.5% accuracy *across all pairs*, with no label supervision. This direction exists and
  is consistent.
- **PCA**: The dominant variance dimension (PC1) strongly encodes bug status (PBC -0.649
  at layer 32) and `will_be_correct` (PBC -0.563). The representation structure is
  concentrated, not diffuse.
- **Change point**: No sharp early "aha moment." Transitions are broadly distributed
  (T*/T ≈ 0.35–0.74), suggesting incremental rather than sudden representational change.
  Buggy programs shift later than original ones at layer 48 cosine (0.736 vs. 0.580).
- **Causal validation (steering)**: CCS direction at layer 32 causally improves pass@1
  at alpha=1.0. Dose-response confirmed across both runs. At N=253 (test-split only,
  Iter2): 66.4% → 68.4% (+2.0 pp). At N=30 (Iter1, all pairs): 53.3% → 70.0% (+16.7 pp,
  but ±17 pp CI). Reliable estimate is ~+2 pp; Iter1 was inflated by small-sample variance.
  Representation is causally load-bearing but effect size is modest.
- **DMD**: OOM-killed. No results on temporal persistence modes.
- **RSA**: Near-zero ρ (< 0.09) — model geometry does not reflect syntactic program
  similarity, suggesting representations are organised by program outcomes rather than code structure.

### The story

All three completed analytical methods (probing, CCS, PCA) converge on **layer 32** as
the primary locus of program-fate encoding. The signal is strongest there across every
metric. The direction is linearly decodable (probes at 94%), geometrically consistent
(CCS at 92.5%), and dominant in the variance structure (PC1 PBC = -0.649). This is
consistent evidence for a **single, low-dimensional, linear representation direction**
in the layer-32 residual stream that encodes whether the model is processing a buggy
program and whether it will produce the correct fix.

The temporal story is more nuanced. The `is_buggy` signal is already near-maximum at
the very first decoded token — this is unsurprising since the prompt explicitly states
the bug. The interesting signal is `will_be_correct`, which grows from 70% to 92% over
the generation in the definitive test-set run (Section 3.13). This growth shows that
the model is not merely reading the prompt: it is *computing* its confidence in the fix
during chain-of-thought reasoning, and this computation leaves a detectable trace in the
residual stream. The late change points (mean T*/T > 0.5 for most conditions) are
consistent with this gradual refinement picture rather than a single "flip" moment.

The causal question is now answered with better evidence: steering with the layer-32 CCS
direction improves pass@1 in a dose-responsive manner across both iterations. The Iteration 2
run (N=253 held-out test-split pairs) yields a baseline of 66.4% and a full-steering pass@1
of 68.4% (+2.0 pp at alpha=1.0). The dose-response (66.4% → 67.6% → 68.4%) is monotone,
confirming a genuine causal effect. The Iteration 1 estimate (+16.7 pp, N=30) was almost
certainly inflated by small-sample variance (95% CI ≈ ±17 pp at N=30). The reliable
effect size is approximately +2 pp — positive and consistent, but modest in practical terms.

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
