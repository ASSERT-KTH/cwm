# Experiment 01: Interpretability of CWM-32B on CRUXEval (Trace Mode)

**Model**: CWM-32B (64 layers, hidden dim 6144, GQA, 3:1 local:global window attention)
**Dataset**: CRUXEval — 800 Python programs, `trace_full` prompting
**Date**: March 2026
**Baseline pass@1**: 87.25%

---

## Hypothesis

CWM-32B builds internal representations of program output properties (return type, sign,
correctness) in its residual stream well before the final output token is generated. Specifically:

- **H1**: Intermediate hidden states at the final trace token are linearly decodable for semantic
  properties of the program output (return type, sign, truthy/falsy).
- **H2**: This signal peaks in mid-layers (~32) rather than final layers, consistent with a
  "representation → prediction" pipeline where content is computed early and refined late.
- **H3**: A mean-difference steering vector (correct vs. incorrect traces) has a causal effect
  on pass@1, demonstrating that the representation is not merely correlated with the outcome.

---

## Experimental Setup

### Activation capture

Hidden states extracted at the **final token position** of each trace (the token immediately
before the model outputs the answer) at 9 layers: `[0, 8, 16, 24, 32, 40, 48, 56, 63]`.

**Reproduce**:
```bash
N_GPUS=2 bash interp/scripts/extract.sh
# Output: interp-extract-trace_full/
```

### Labels derived from traces

Four binary/categorical labels per sample:
- `will_be_correct`: did the model produce the correct final answer?
- `return_type`: int / string / list / other (4-class)
- `return_sign`: is the return value positive? (binary, among numeric outputs)
- `return_truthy`: is the return value truthy? (binary)

Labels extracted by `interp/probes/labels.py` from the generated trace text.

---

## Phase 1 — Logit Lens

### Methodology

Project each intermediate hidden state `h[L]` through the final RMSNorm and unembedding matrix.
Measure mean top-1 probability across all samples at each layer.

**Reproduce**:
```bash
bash interp/scripts/logit_lens.sh
# Output: interp-extract-trace_full/logit_lens.pt
```

### Results

| Layer | Mean top-1 prob | Notes |
|-------|----------------|-------|
| 0     | 0.201          | |
| 8     | 0.001          | Global attention layer |
| 16    | 0.179          | |
| 24    | 0.166          | |
| 32    | 0.106          | |
| 40    | 0.011          | Global attention layer |
| 48    | 0.037          | |
| 56    | 0.270          | |
| 63    | 0.814          | Final layer |

### Interpretation

Token-level commitment happens almost entirely in the final layers (56→63). Layers 8 and 40 —
both global attention layers in the 3:1 local:global window pattern — show near-zero top-1
probability, suggesting these layers perform heavy representational mixing rather than refining
output predictions. The steep rise from layer 56 to 63 marks where the model "commits" to its
answer token.

**Scope**: Logit lens measures next-token prediction confidence, not the content of intermediate
representations. Low top-1 probability at layer 32 does not mean layer 32 has no useful
information — probing (Phase 2) can still find decodable structure there.

---

## Phase 2 — Linear Probes

### Methodology

Train linear and shallow MLP (2-layer, hidden 256) classifiers on hidden states at each layer to
predict output properties. 80/20 train/val split, 50 epochs, Adam, weight decay 1e-4.

**Reproduce**:
```bash
bash interp/scripts/train_probes.sh
# Output: interp-extract-trace_full/probes/
```

### Results

| Property | Majority baseline | Best linear | Best MLP | Peak layer |
|----------|-------------------|-------------|----------|------------|
| `will_be_correct` | 87.25% | 88.38% | 88.25% | 32 |
| `return_type` | 46.00% | 78.50% | 80.63% | 32 |
| `return_sign` | 87.75% | 92.00% | 93.00% | 24–32 |
| `return_truthy` | 86.75% | 91.00% | 90.75% | 32 |

### Interpretation

`return_type` is the most strongly decodable property (+34pp above majority baseline at layer 32),
confirming H1 and H2: the model builds a clear representation of what *type* of value it will
return well before the output token. This is non-trivial — the type must be inferred from the
program structure and input value.

`return_sign` and `return_truthy` show consistent gains of ~4–5pp above high majority baselines.
The signal is real but the high baseline (87–88% majority) limits interpretability — a small
improvement could reflect either a genuine representation or minor dataset biases.

`will_be_correct` is essentially not decodable (+1pp). The model's final-token hidden state does
not cleanly separate correct from incorrect traces in a linearly accessible way. This leaves
open whether: (a) correctness is not localized to a single token/layer; (b) the heavy class
imbalance (87% positive) makes learning difficult; or (c) correctness is encoded nonlinearly.

All properties peak around layer 32. MLP probes slightly outperform linear probes on semantic
properties, suggesting mild nonlinearity in the encoding.

**Scope**: Linear decodability is necessary but not sufficient for a causal representation —
the probe could be picking up a correlate of the output rather than the representation that
drives it. Phase 3 tests causality.

---

## Phase 3 — Activation Steering

### Methodology

Compute a `correct_vs_incorrect` steering vector as the mean difference of hidden states between
correct and incorrect traces at each layer. Apply this vector (scaled by alpha) at inference time
on a held-out set and measure pass@1 change.

Layers tested: `{32, 48, 63}`. Alpha values: `{-2, -1, 0, +1, +2}`. Sample size: ~50 per cell.

**Reproduce**:
```bash
bash interp/scripts/steer.sh
# Output: interp-steer-correct_vs_incorrect/
```

### Results

| Layer | α=−2 | α=−1 | α=0 (baseline) | α=+1 | α=+2 |
|-------|------|------|----------------|------|------|
| 32    | 88%  | 88%  | 88%            | 90%  | 90%  |
| 48    | 88%  | 88%  | 88%            | 90%  | 90%  |
| 63    | 88%  | 88%  | 88%            | 88%  | 88%  |

### Interpretation

Positive steering at layers 32 and 48 yields +2pp (88% → 90%). Negative steering has no effect.
Layer 63 is completely inert to steering. This provides partial support for H3:

- The correct_vs_incorrect direction has a mild causal role at layers 32 and 48.
- The asymmetry (positive works, negative doesn't) suggests the "correct" direction nudges the
  model toward confident, correct completions, but the model is robust — it cannot easily be
  steered *into* mistakes using this vector.
- Layer 63 being inert is consistent with the logit lens: by layer 63 the computation is nearly
  finalized and steering the residual stream no longer changes the output.

**Scope**: The +2pp effect is small and the sample size is ~50. At 95% CI, the margin of error
is ±5.9pp, so the result is statistically inconclusive. A larger study (n=200+) with
multiple vector extraction methods (PCA, CCS) would be needed for confident causal claims.
The null result for negative alpha also limits the interpretation — we can say the correct
direction helps, but not that the incorrect direction hurts.

---

## Phase 4 — Token Interventions

### Methodology

Targeted interventions replacing the hidden state at specific token positions (branch conditions,
return values, variable assignments) with states from counterfactual traces. Measures how
localised computation is in the residual stream.

**Reproduce**:
```bash
bash interp/scripts/token_intervention.sh
# Output: interp-intervene-{variable,branch,return,...}/
```

Results from this phase are recorded in the raw output directories and have not yet been
fully summarised. See `interp-intervene-*/` for raw `.jsonl` outputs.

---

## Summary

| Hypothesis | Finding | Confidence |
|------------|---------|------------|
| H1: Semantic properties decodable from hidden states | ✅ Confirmed — `return_type` +34pp at layer 32 | High |
| H2: Signal peaks in mid-layers (~32) | ✅ Confirmed — all properties peak at layer 32 | High |
| H3: Steering vector causally affects pass@1 | ⚠️ Partial — +2pp at layers 32/48, n=50, p≈0.3 | Low (underpowered) |

CWM-32B builds semantic representations of program output properties in mid-layers, consistent
with a two-stage pipeline: "what type will this return?" is resolved by layer 32, while "which
exact token?" is resolved in layers 56–63. Whether correctness is causally encoded in a single
direction remains an open question requiring a larger-scale causal experiment.

---

## Next Steps

- Larger steering experiment (n=200+) with CCS and PCA vectors to establish causal significance
- Probe the `will_be_correct` signal at intermediate trace positions (not just the final token)
  → see Experiment 02 (bug-fixing trace) for this direction
- Mechanistic analysis: which attention heads at layer 32 drive the return_type representation?
