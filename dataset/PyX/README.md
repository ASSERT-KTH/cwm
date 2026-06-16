# PyX → CRUXEval format

Converts [`semcoder/PyX`](https://huggingface.co/datasets/semcoder/PyX) into the
same 4-field schema as CRUXEval-O (`{id, code, input, output}`), for the
"execute → collect trace" pipeline (CODI distillation / Table 9 style).

## Source

- HF: [`semcoder/PyX`](https://huggingface.co/datasets/semcoder/PyX), split
  `train` (93,456 rows). Fields: `id, nl, fwd_mnl, bwd_mnl, response`.
- Paper: Ding et al., *SemCoder: Training Code Language Models with
  Comprehensive Semantics Reasoning*, NeurIPS 2024.

PyX mixes plain NL→code rows (empty `fwd_mnl`/`bwd_mnl`) with CRUXEval-style
input/output-prediction rows. We keep **only** the latter: a row qualifies when
its monologue field carries a `[PYTHON]` block (the function under test) and its
`response` carries an `[ANSWER]` block with a concrete
`assert f(args) == expected`. About **75%** of rows qualify; the plain NL→code
rows have no concrete input and cannot be traced, so they are dropped.

## Conversion method (see [convert.py](convert.py))

- **code** = the `[PYTHON]` block with its trailing probe `assert` cut off (the
  probe contains `??`, masking the predicted side, so it is stripped textually
  before parsing). An alias `f = <entry>` is appended; the trace wrapper then
  runs `def main(): return f({input})`.
- **input / output** = parsed from the `[ANSWER]` `assert <entry>(args) == expected`
  via AST. Unlike MBPP/HumanEval, keyword/`*`/`**` arguments are **kept** and
  faithfully reconstructed (e.g. `10, b=5, c=5, d=8`), since PyX inputs use them.
- **entry** = the function called in the `[ANSWER]` assert.
- **id** = `pyx_<source id>` (e.g. `pyx_fwd_mnl_108213_ipt9`).

`output` is recorded for reference only; the trace pipeline recomputes the
return by execution. On a 320-row sample, 100% of qualifying rows executed
cleanly and the computed return matched the recorded `output` exactly.

## Usage

```bash
python convert.py   # download semcoder/PyX and save_to_disk into ./data
```

```python
from dataset.sources import load_rows
rows = load_rows(["pyx"])                 # or ["pyx", "mbpp", "humaneval"]
```

`./data` is a 4-column HF Dataset isomorphic to CRUXEval-O. Train/val split and
CODI tokenization live in the consumer `cwm.training.data.build_dataset`.

## Caveats

- Only rows with a `[PYTHON]`+`[ANSWER]` pair are kept; plain NL→code rows are
  dropped.
- Self-contained single/recursive functions; blocks ship their own imports.
- Downstream `build_example` drops empty traces or examples over `max_seq_len`;
  this script does no trace-length filtering of its own.
