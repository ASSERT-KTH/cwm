# Table 9 reproduction — execution-trace prediction analysis

Reproduces **Table 9** of *CWM: An Open-Weights LLM for Research on Code
Generation with World Models* ([arXiv:2510.02387](https://arxiv.org/abs/2510.02387)):
the component-wise breakdown of CWM's **full execution-trace prediction**,
evaluated with **greedy decoding**.

Modeled on [`evals/cruxeval`](../cruxeval): it reuses the same full-trace prompt
(`make_trace_full_prompt_tokens`), the FastGen/ImpGen distributed inference
stack, and the CRUXEval dataset. Where `cruxeval` scores only the final output
(pass@1), this eval also parses the whole predicted trace and compares it
frame-by-frame against a ground-truth trace.

## Paper reference (Table 9)

|                | metric                     | CruxEval | Function-level |
|----------------|----------------------------|---------:|---------------:|
| Output         | pass@1                     | 88.0     | 94.4           |
| Trace          | Valid Trace Format         | 99.6     | 100.0          |
|                | State Exact Match          | 96.9     | 96.4           |
|                | Action Exact Match         | 96.5     | 98.0           |
| States         | Valid JSON Format          | 100.0    | 100.0          |
|                | Key Match                  | 99.1     | 99.0           |
|                | Key+Value Match            | 98.1     | 97.9           |
| Statistics     | Avg State Length (Token)   | 11.7     | 18.8           |
|                | Avg Action Length (Token)  | 11.2     | 10.0           |

Only the **CruxEval** column is reproducible from open data; the
**Function-level** column uses Meta-internal data and is omitted.

## What each metric means

Per sample we prompt CWM in full-trace mode, parse the generation into frames,
build the ground-truth trace by executing the function under `sys.settrace`, and
align the two traces positionally. Rates are macro-averaged over traces (mean of
per-trace fractions), matching the paper's "… per execution trace" phrasing.

- **Output pass@1** — extract the final return value and execute
  `assert <expected> == <predicted>` (reuses `cruxeval.evaluate`).
- **Valid Trace Format** — fraction of generations that parse into well-formed
  frames (event token + `<|action_sep|>`, plus `<|arg_sep|>` for return/exception
  frames, no trailing garbage before `<|end_of_text|>`).
- **State Exact Match** — fraction of GT observation states whose predicted
  locals dict matches exactly (diff-based representation).
- **Action Exact Match** — fraction of GT actions (source lines) reproduced exactly.
- **Valid JSON Format** — fraction of predicted states that parse as a JSON object.
- **Key / Key+Value Match** — mean per-state Jaccard overlap of variable names /
  `(name, value)` pairs.
- **Avg State / Action Length (Token)** — average token length of a state payload
  / action line over the ground-truth traces.

## Trace format

See [`PROMPTING_GUIDE.md`](../../PROMPTING_GUIDE.md) and
[`demos/cwmdbg.py`](../../demos/cwmdbg.py). A frame is one of:

```
<|call_sep|>$LOCALS<|action_sep|>$SOURCE<|frame_sep|>
<|line_sep|>$LOCALS<|action_sep|>$SOURCE<|frame_sep|>
<|return_sep|><|action_sep|>$SOURCE<|arg_sep|>$VALUE<|frame_sep|>
<|exception_sep|><|action_sep|>$SOURCE<|arg_sep|>$VALUE<|frame_sep|>
```

`$LOCALS` is a JSON object of `name -> repr(value)` strings, using a
**diff-based** representation (a variable unchanged since the previous frame in
the same scope renders as the placeholder `".."`). Values use Python `repr`
semantics — `'x'`, `(4, 1)` — *not* `json.dumps` (`"x"` / `[4, 1]`); the repr
string is stored as a JSON string value. The entry point is a synthetic
`def main(): return f(<input>)`, and the prompt **seeds** the first
`call main()` frame, so the model generates from the second frame onward; the
ground-truth tracer drops that seeded frame before alignment.

## Files

| file               | purpose |
|--------------------|---------|
| `trace_format.py`  | Frame dataclass; parse a generation into frames; render frames back. |
| `ground_truth.py`  | `sys.settrace`-based ground-truth tracer in CWM's frame format. |
| `metrics.py`       | Per-trace Table 9 components + macro aggregation + table formatter. |
| `run_eval.py`      | Distributed driver (greedy full-trace generation + scoring). |
| `score.py`         | Offline re-scoring of a saved `results.jsonl` (no GPU). |
| `eval.sbatch`      | 8-GPU Slurm job over the full CRUXEval test split. |
| `smoke_test.sbatch`| 4-GPU, 8-sample smoke test. |

## Running

Full run (8 GPUs, 4 TP groups of 2 → 4-way data parallelism):

```bash
python -m torch.distributed.run --nproc_per_node=8 \
    -m evals.trace_analysis.run_eval \
    checkpoint_dir=/path/to/cwm \
    gen_args.tp_size=2 \
    dump_dir=./eval-cwm-table9
```

or `sbatch evals/trace_analysis/eval.sbatch`. Greedy decoding is the default
(`gen_args.use_sampling=False`). Results land in `dump_dir/results.jsonl`, the
aggregate in `dump_dir/summary.json`, and a formatted table is printed.

Re-score offline (e.g. after changing the tracer), no GPU:

```bash
python -m evals.trace_analysis.score ./eval-cwm-table9/results.jsonl \
    --tokenizer /path/to/cwm/tokenizer.model
```

Unit tests (GPU-free): `pytest tests/trace_analysis/`

## Known gaps vs. the paper

A faithful **re-implementation**, not a bit-exact replica of Meta's internal
tracer; expect small deviations from the published numbers.

1. **Validation vs. test split.** The paper evaluates on CRUXEval *validation*
   inputs; the public `cruxeval-org/cruxeval` dataset only ships a `test` split
   (800 samples), which is what we use (as `cruxeval/run_eval` does).
2. **Function-level column** uses internal data and is omitted.
3. **Diff-reset rules.** CWM's exact scope/diff-reset semantics are unpublished;
   we reset the diff per call frame and treat a value as unchanged when its
   rendered string matches the previous frame's in the same scope. Exotic objects
   may also differ from CWM's training-time renderer.
4. **Frame alignment** is positional: a well-formed prediction aligns exactly; a
   malformed/truncated one is scored against the shorter common prefix.

Because of (3), `State Exact Match` / `Key+Value Match` are the metrics most
sensitive to renderer mismatch; `Valid Trace Format`, `Valid JSON Format`, and
`Action Exact Match` are robust. If value-sensitive rows are systematically low,
inspect a few dumped `generation` fields against `ground_truth_trace(...)` and
adjust `render_value` / the diff rule in `ground_truth.py`, then re-run `score.py`.
