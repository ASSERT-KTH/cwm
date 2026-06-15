# HumanEval → CRUXEval format

Converts HumanEval into the same 4-field schema as CRUXEval-O
(`{id, code, input, output}`), for the "execute → collect trace" pipeline
(CODI distillation / Table 9 style).

## Source

- HF: [`openai/openai_humaneval`](https://huggingface.co/datasets/openai/openai_humaneval) (test, 164 problems)
- Upstream repo: <https://github.com/openai/human-eval>
- Paper: Chen et al., *Evaluating Large Language Models Trained on Code*, 2021
- License: MIT

## Conversion method (see [convert.py](convert.py))

Each source problem expands into one or more rows (one per assertion):

- **code** = `prompt + canonical_solution`, with an alias `f = <entry_point>`
  appended (no renaming, to avoid breaking recursion / substrings). The trace
  wrapper then runs `def main(): return f({input})`.
- **input / output**: parse each assertion inside `check(candidate)` from the
  `test` field with `ast`, keeping only single `==` comparisons of the form
  `candidate(args) == <expected>` (or reversed):
  - `input` = source segments of the call args joined by `, ` → spliced into `f(input)`;
  - `output` = source segment of the other side of the comparison.
  - Non-`==` forms (keyword args, `abs(...) < eps` float tolerance, `in`,
    boolean asserts) are skipped.
- **id** = `<task_id with '/' replaced by '_'>_<assert index>`.

## Usage

```bash
python convert.py   # download and save_to_disk into ./data
```

```python
from dataset.sources import load_rows
from cwm.training.data import build_dataset

rows = load_rows(["humaneval"])                              # or ["mbpp", "humaneval"]
examples = build_dataset(rows, tokenizer, max_seq_len=8192)  # -> [(input_ids, labels), ...]
```

`./data` is a 4-column HF Dataset isomorphic to CRUXEval-O. `dataset/` only
switches sources (`dataset.sources.load_rows`) and generates traces
(`dataset.ground_truth` / `dataset.trace_format`); train/val split and CODI
tokenization live in the consumer `cwm.training.data.build_dataset`.

## Caveats

- Only clean `==` assertions are kept; the rest are dropped.
- Downstream `build_example` automatically drops empty traces or examples longer
  than `max_seq_len`; this script does no trace-length filtering of its own.
