# Table 9 reproduction — run record

Full-split run of [`evals/trace_analysis`](../evals/trace_analysis) reproducing
Table 9 (CruxEval column) of *CWM* ([arXiv:2510.02387](https://arxiv.org/abs/2510.02387)).
Date: 2026-06-01.

## Config

| | |
|---|---|
| Slurm job | `16695294` (node073), `sbatch evals/trace_analysis/eval.sbatch` |
| Hardware | 1 node × 8 A100-SXM4-80GB, `tp_size=2` → 4-way data parallel |
| Account | `berzelius-2026-167` |
| Model | `./model_weights/cwm` |
| Decoding | greedy (`gen_args.use_sampling=False`) |
| Dataset | `cruxeval-org/cruxeval` test split, n=800 |
| Code | `cwm_andre` submodule @ `c42478e` |
| Wall time | ~1h35m (bounded by a few very long traces in dp=0) |

Command (as run by the sbatch script):

```bash
python -m torch.distributed.run --nproc_per_node=8 \
    -m evals.trace_analysis.run_eval \
    checkpoint_dir=./model_weights/cwm \
    gen_args.tp_size=2 \
    dump_dir=./eval-cwm-table9
```

## Results vs. paper (Table 9, CruxEval column)

| metric | paper | this run (n=800) | Δ |
|---|---:|---:|---:|
| Output pass@1 | 88.0 | 87.25 | −0.75 |
| Valid Trace Format | 99.6 | 99.63 | +0.03 |
| State Exact Match | 96.9 | 96.41 | −0.49 |
| Action Exact Match | 96.5 | 98.68 | +2.18 |
| Valid JSON Format | 100.0 | 99.55 | −0.45 |
| Key Match | 99.1 | 99.55 | +0.45 |
| Key+Value Match | 98.1 | 97.48 | −0.62 |
| Avg State Length (Token) | 11.7 | 14.29 | +2.59 |
| Avg Action Length (Token) | 11.2 | 9.21 | −1.99 |

All 7 quality metrics land within ±2.2 pt (most <1 pt) — close agreement for an
independent re-implementation. The two **Avg Length** rows are dataset
statistics (token distribution of the traces), not correctness signals: they
shift with the val-vs-test split (paper uses internal *validation* inputs; the
public dataset only ships *test*) and with tracer rendering details. The
`Function-level` column of the paper uses Meta-internal data and is not
reproducible here.

## Files

| file | contents |
|---|---|
| `summary.json` | aggregate metrics + `n`, `decoding` |
| `results.jsonl` | 800 rows; each has `code`, `input`, `generation`, per-sample scores |
| `results_dp{0..3}.jsonl` | per-data-parallel-rank shards (merged into `results.jsonl`) |

Re-score offline without a GPU (e.g. after changing the tracer):

```bash
python -m evals.trace_analysis.score ./eval-cwm-table9/results.jsonl \
    --tokenizer ./model_weights/cwm/tokenizer.model
```
