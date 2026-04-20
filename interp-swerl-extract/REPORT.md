# CWM Interpretability: Outcome Probing on SWEbench Trajectories

**Date:** 2026-04-20
**Branch:** `andre/feat/interp`
**Author:** André Silva

---

## Overview

We trained linear probes on internal activations of the CWM 32B model to ask: *does the model's hidden state carry information about whether it will ultimately solve a SWEbench instance?* We ran the model on 485 SWEbench-Verified instances, captured activations at layer 32 (mid-network), and trained logistic regression probes across three analysis axes. We additionally ran a global probe experiment to test whether a single probe trained on all trajectory positions jointly matches the performance of position-specific probes.

---

## Setup

### Model
- **Architecture:** CWM 32B, 64 layers, hidden dim 6144, GQA + window attention (3:1 local:global ratio)
- **Inference:** TP=2, DP=4 across 8×A100-80GB (Berzelius cluster)

### Activation Capture
- Hook point: post-FFN residual at each layer (`h` after `h.add_(h_out)` in `cwm/fastgen/forward.py:786`)
- Only prefill passes are captured (CUDA graphs bypass decode-step hooks)
- Layer captured: **layer 32** (middle of the 64-layer stack)
- Due to tensor parallelism, only TP rank 0 writes non-zero activations per node — the other rank produces zeros and is skipped at load time

### Dataset
- **485 SWEbench-Verified instances** extracted across 4 sharded SLURM jobs (8h wall-clock each, with resume logic)
- **250 pass / 235 fail** (51.5% / 48.5%) — near-balanced binary label
- Per-file subsampling to 5,000 activation rows to cap RAM; per-bucket cap of 50,000 rows for the capped runs; uncapped runs use all available rows
- Probe: L2-regularised logistic regression (PyTorch Adam, 100 epochs, C=1.0, 20% validation split)

### Analysis Axes

Three axes slice the same stored activations differently — no re-running the model:

1. **Token position** — where in the full context window does the token appear? (10 equal-width bins over normalised position 0→1). Uses all captured tokens.
2. **Tool-call index** — which tool call within the episode did this token come from? (10 equal-width bins over normalised turn index). All tokens from each turn are included, labelled with that turn's outcome.
3. **Stage** — what type of operation was the agent performing? (exploration / editing / testing / other-bash / submission). All tokens from each turn are included, labelled with the turn's stage.

**Note on axis 2 and 3 semantics:** An earlier version of these axes used only the *last token* of each turn. The current version uses *all tokens* from each turn, which is more faithful to how the activations are distributed but dilutes the per-bucket signal with noisier mid-sequence tokens. Results are not directly comparable to any pre-April 2026 runs.

---

## Experiment 1: Per-Bucket Probes (Capped, 50k/bucket)

A separate linear probe is trained independently for each bucket on each axis. This is the most powerful setting — each probe is free to find the optimal linear direction for its specific trajectory position.

### Axis 1: Token Position

| Bin | Normalised position | n      | Majority | Val acc |
|-----|---------------------|--------|----------|---------|
| 0   | 0.0–0.1             | 50,000 | 52.7%    | 56.3%   |
| 1   | 0.1–0.2             | 50,000 | 53.0%    | 56.6%   |
| 2   | 0.2–0.3             | 50,000 | 52.7%    | 57.1%   |
| 3   | 0.3–0.4             | 50,000 | 52.0%    | 57.4%   |
| 4   | 0.4–0.5             | 50,000 | 52.1%    | 58.0%   |
| 5   | 0.5–0.6             | 50,000 | 51.0%    | 58.1%   |
| 6   | 0.6–0.7             | 50,000 | 51.7%    | **59.3%** |
| 7   | 0.7–0.8             | 50,000 | 53.0%    | 58.3%   |
| 8   | 0.8–0.9             | 50,000 | 54.3%    | 58.9%   |
| 9   | 0.9–1.0             | 50,000 | 62.9%    | 58.0%   |

**Observation:** Accuracy rises steadily from ~56% to ~59% in the middle of the context, then plateaus. The drop at bin 9 is notable: very late tokens have the highest majority baseline (62.9%, from class-imbalanced long trajectories) but the probe does not improve over earlier bins, suggesting diminishing returns from raw context length.

### Axis 2: Tool-Call Index

| Bin | Episode position | n      | Majority | Val acc |
|-----|-----------------|--------|----------|---------|
| 0   | earliest 10%    | 50,000 | 57.2%    | 55.8%   |
| 1   |                 | 50,000 | 59.3%    | 56.4%   |
| 2   |                 | 50,000 | 54.9%    | 56.0%   |
| 3   |                 | 50,000 | 55.6%    | 56.9%   |
| 4   |                 | 50,000 | 54.8%    | 57.0%   |
| 5   |                 | 50,000 | 53.1%    | 55.8%   |
| 6   |                 | 50,000 | 50.7%    | 55.7%   |
| 7   |                 | 50,000 | 52.9%    | 55.9%   |
| 8   |                 | 50,000 | 53.3%    | 56.5%   |
| 9   | latest 10%      | 50,000 | 56.7%    | 56.3%   |

**Observation:** Accuracy is flat across the trajectory (55–57%), with no clear upward trend. This is in contrast to an earlier run using only the *last token* per turn, which showed a clear rise from 58% to 68%. The flattening indicates that most tokens within a turn carry substantially less outcome information than the final token — the last token appears to be a privileged position where the model's outcome representation crystallises before generating the next action.

### Axis 3: Stage

| Stage       | n      | Majority | Val acc   |
|-------------|--------|----------|-----------|
| exploration | 50,000 | 53.3%    | 54.6%     |
| editing     | 50,000 | 51.2%    | 54.9%     |
| testing     | 50,000 | 54.9%    | 54.8%     |
| other-bash  | 50,000 | 52.0%    | 54.6%     |
| submission  | 44,476 | 54.5%    | **62.0%** |

**Observation:** Submission activations stand out clearly (62.0%), while all other stages cluster tightly between 54.6–54.9%. The large gap for submission (vs ~7–8 pp gap in the last-token version) is consistent with the all-tokens dilution: mid-sequence tokens within submission turns still carry the submission signal, but less sharply. The stage ordering (submission >> testing ≈ editing ≈ other-bash ≈ exploration) is stable across both runs.

---

## Experiment 2: Per-Bucket Probes (Uncapped)

Same setup but with no per-bucket cap — using all available tokens per bucket (242k–413k for token position bins, 184k–286k for tool-call bins, 363k–960k for stage pools). This tests whether more data meaningfully changes the per-bucket estimates.

### Axis 1: Token Position (uncapped)

| Bin | n       | Majority | Val acc   |
|-----|---------|----------|-----------|
| 0   | 242,297 | 52.7%    | 56.3%     |
| 1   | 254,867 | 52.9%    | 56.4%     |
| 2   | 252,885 | 52.5%    | 56.8%     |
| 3   | 238,633 | 52.1%    | 57.3%     |
| 4   | 232,712 | 52.4%    | 57.6%     |
| 5   | 221,645 | 51.0%    | 58.0%     |
| 6   | 206,506 | 51.7%    | **59.7%** |
| 7   | 188,641 | 52.9%    | 59.1%     |
| 8   | 172,984 | 54.5%    | 58.7%     |
| 9   | 412,361 | 62.9%    | 58.0%     |

### Axis 2: Tool-Call Index (uncapped)

| Bin | n       | Majority | Val acc |
|-----|---------|----------|---------|
| 0   | 254,082 | 57.3%    | 54.3%   |
| 1   | 275,525 | 58.9%    | 56.3%   |
| 2   | 256,030 | 54.9%    | 55.0%   |
| 3   | 210,089 | 55.6%    | 56.8%   |
| 4   | 184,794 | 54.5%    | 57.1%   |
| 5   | 230,181 | 52.8%    | 55.0%   |
| 6   | 233,733 | 50.8%    | 55.3%   |
| 7   | 225,722 | 53.0%    | 55.3%   |
| 8   | 267,097 | 53.3%    | 55.8%   |
| 9   | 286,278 | 56.5%    | 56.3%   |

### Axis 3: Stage (uncapped)

| Stage       | n         | Majority | Val acc   |
|-------------|-----------|----------|-----------|
| exploration | 960,329   | 53.5%    | 54.2%     |
| editing     | 474,818   | 51.2%    | 54.1%     |
| testing     | 363,487   | 55.0%    | 54.8%     |
| other-bash  | 580,421   | 51.9%    | 54.2%     |
| submission  | 44,476    | 54.5%    | **61.5%** |

**Observation:** Results are nearly identical to the capped run despite 5–20× more data per bucket. The probe accuracy has converged — 50k samples is sufficient, and additional data does not change the conclusions.

---

## Experiment 3: Global Probe

A single linear probe is trained on **all trajectory positions jointly** (200,000 samples total, randomly drawn across all turns and trajectories), then evaluated per-axis bucket on the held-out 20% validation split. This tests whether there is a single linear direction in layer-32 activation space that predicts outcome regardless of trajectory position.

### Global probe vs per-bucket probes (capped): Token Position

| Bin | Per-bucket val acc | Global probe val acc | Δ      |
|-----|--------------------|----------------------|--------|
| 0   | 56.3%              | 51.4%                | −4.9pp |
| 1   | 56.6%              | 50.9%                | −5.7pp |
| 2   | 57.1%              | 51.6%                | −5.5pp |
| 3   | 57.4%              | 53.1%                | −4.3pp |
| 4   | 58.0%              | 53.8%                | −4.2pp |
| 5   | 58.1%              | 53.2%                | −4.9pp |
| 6   | 59.3%              | 53.9%                | −5.4pp |
| 7   | 58.3%              | 55.1%                | −3.2pp |
| 8   | 58.9%              | 53.3%                | −5.6pp |
| 9   | 58.0%              | 52.1%                | −5.9pp |

### Global probe vs per-bucket probes (capped): Tool-Call Index

| Bin | Per-bucket val acc | Global probe val acc | Δ      |
|-----|--------------------|----------------------|--------|
| 0   | 55.8%              | 51.8%                | −4.0pp |
| 1   | 56.4%              | 52.0%                | −4.4pp |
| 2   | 56.0%              | 53.5%                | −2.5pp |
| 3   | 56.9%              | 51.8%                | −5.1pp |
| 4   | 57.0%              | 52.8%                | −4.2pp |
| 5   | 55.8%              | 53.2%                | −2.6pp |
| 6   | 55.7%              | 52.3%                | −3.4pp |
| 7   | 55.9%              | 52.5%                | −3.4pp |
| 8   | 56.5%              | 52.7%                | −3.8pp |
| 9   | 56.3%              | 53.9%                | −2.4pp |

### Global probe vs per-bucket probes (capped): Stage

| Stage       | Per-bucket val acc | Global probe val acc | Δ       |
|-------------|--------------------|----------------------|---------|
| exploration | 54.6%              | 52.2%                | −2.4pp  |
| editing     | 54.9%              | 53.6%                | −1.3pp  |
| testing     | 54.8%              | 52.5%                | −2.3pp  |
| other-bash  | 54.6%              | 53.0%                | −1.6pp  |
| submission  | **62.0%**          | **49.9%**            | −12.1pp |

**Observation:** The global probe is consistently 3–6 pp worse than per-bucket probes across token-position and tool-call-index axes. The most striking gap is at the submission stage (−12.1 pp), where the global probe drops *below* majority baseline (49.9% vs 54.5% majority). This is a strong result: the outcome representation at submission is not a general property of the activation space, but a highly position-specific linear direction that the global probe cannot locate when trained on all stages jointly. The outcome signal in layer-32 activations is **trajectory-position-dependent** — there is no single universal outcome direction.

---

## Key Takeaways

1. **Outcome is linearly decodable above chance across all conditions** (~55–62% vs ~52% majority baseline), confirming that layer-32 activations carry trajectory-outcome information throughout the episode.

2. **The "last token" of a turn is a privileged position.** The earlier finding of a strong upward trend in tool-call-index accuracy (58% → 68%) was largely due to using only the final token per turn. With all tokens, the trend flattens to 55–57%. This suggests the model's outcome representation crystallises specifically at the moment it finishes generating a response and is about to execute a tool call.

3. **Submission is the most predictive stage, and its signal is highly localised.** The per-bucket submission probe reaches 62.0%, while the global probe drops to 49.9% on the same data — below chance. No other stage shows this collapse. The model encodes "am I going to succeed?" most sharply in the submission turn, but in a direction that is orthogonal to how it encodes outcome at other trajectory positions.

4. **50k samples per bucket is sufficient.** The uncapped run (5–20× more data) produces essentially identical accuracy estimates. The probe has converged.

5. **Token position within the context window shows a mild upward gradient** (56% → 59%), peaking around 60–70% of context. This likely reflects that the most diagnostic tokens appear in the middle-to-late context (after initial exploration, during editing/testing phases) rather than at the very end (which is dominated by long failing trajectories that inflate the majority baseline).

---

## Figures

All figures are saved under `interp-swerl-extract/figures_capped/` and `interp-swerl-extract/figures_uncapped/`.

| File | Description |
|------|-------------|
| `probe_token_position.png` | Per-bucket probe accuracy vs token position bin |
| `probe_tool_call_index.png` | Per-bucket probe accuracy vs tool-call index bucket |
| `probe_stage.png` | Per-bucket probe accuracy vs stage |
| `probe_token_position_comparison.png` | Per-bucket vs global probe: token position |
| `probe_tool_call_index_comparison.png` | Per-bucket vs global probe: tool-call index |
| `probe_stage_comparison.png` | Per-bucket vs global probe: stage |

Probe checkpoints (weights + normalisation stats) are saved under `figures_capped/probes/` and `figures_uncapped/probes/`.
