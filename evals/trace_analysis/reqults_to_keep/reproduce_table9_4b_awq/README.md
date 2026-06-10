# Table 9 — 4-bit AWQ run record

Full-split run of [`evals/trace_analysis`](..) on the **4-bit AWQ** CWM via vLLM,
reproducing Table 9 (CruxEval column) of *CWM*
([arXiv:2510.02387](https://arxiv.org/abs/2510.02387)). Date: 2026-06-03.

## Config

| | |
|---|---|
| Slurm job | `16759604` (node087), `sbatch evals/trace_analysis/eval_vllm_awq_4bit.sbatch` |
| Hardware | 1 × A100-SXM4-80GB, `tp_size=1` (model loads in ~19 GiB, true int4) |
| Model | [`cyankiwi/cwm-AWQ-4bit`](https://huggingface.co/cyankiwi/cwm-AWQ-4bit) → `./model_weights/cwm-awq-4bit` (compressed-tensors AWQ int4, Marlin kernel) |
| Runtime | vLLM 0.9.2 (separate conda env), greedy decoding |
| Dataset | `cruxeval-org/cruxeval` test split, n=800 |
| Wall time | 11m44s (incl. ~90s engine init) |

```bash
python -m evals.trace_analysis.run_eval_vllm \
    --model ./model_weights/cwm-awq-4bit \
    --dump_dir ./eval-cwm-table9-awq-4bit --tp_size 1
```

## Results: 4-bit AWQ vs. bf16 baseline vs. paper

| metric | paper | bf16 (n=800) | AWQ-4bit (n=800) |
|---|---:|---:|---:|
| Output pass@1 | 88.0 | 87.25 | 87.00 |
| Valid Trace Format | 99.6 | 99.63 | 99.75 |
| State Exact Match | 96.9 | 96.41 | 96.37 |
| Action Exact Match | 96.5 | 98.68 | 98.69 |
| Valid JSON Format | 100.0 | 99.55 | 99.53 |
| Key Match | 99.1 | 99.55 | 99.53 |
| Key+Value Match | 98.1 | 97.48 | 97.45 |
| Avg State Length (Token) | 11.7 | 14.29 | 14.29 |
| Avg Action Length (Token) | 11.2 | 9.21 | 9.21 |

Int4 AWQ quantization preserves trace-prediction quality: every quality metric is
within ~0.1 pt of the bf16 run and pass@1 within 0.25 pt, while running in ~19 GiB
instead of ~64 GiB. (The bf16 column is the native FastGen run; see
`eval_records/cwm/reproduce_cwm_table9`. Avg-Length rows are dataset statistics,
not correctness signals.)

## Files

| file | contents |
|---|---|
| `summary.json` | aggregate metrics + `n`, `decoding`, `model`, `loader` |
| `results.jsonl` | 800 rows; each has `code`, `input`, `generation`, per-sample scores |

Re-score offline (no GPU):

```bash
python -m evals.trace_analysis.score \
    reproduce_cwm_table9_awq_4bit/results.jsonl \
    --tokenizer model_weights/cwm_ckpt/tokenizer.model
```
