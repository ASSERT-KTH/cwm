# MBPP → CRUXEval format

Converts MBPP into the same 4-field schema as CRUXEval-O
(`{id, code, input, output}`), for the "execute → collect trace" pipeline
(CODI distillation / Table 9 style).

## Source

- HF: [`google-research-datasets/mbpp`](https://huggingface.co/datasets/google-research-datasets/mbpp),
  config `full`, all splits merged (train/test/validation/prompt, 974 problems)
  to maximize yield.
- Paper: Austin et al., *Program Synthesis with Large Language Models*, 2021
- License: CC-BY-4.0

## Conversion method (see [convert.py](convert.py))

Each source problem expands into one or more rows (one per assertion):

- **entry_point**: MBPP gives no explicit function name, so it is taken as the
  function called in the first parseable assertion of `test_list` (via AST).
- **code** = `test_setup_code + "\n" + code`, with an alias `f = <entry_point>`
  appended (no renaming). The trace wrapper then runs `def main(): return f({input})`.
- **input / output**: parse each assertion in `test_list` with `ast`, keeping
  only single `==` comparisons of the form `<entry>(args) == <expected>` (or reversed):
  - `input` = source segments of the call args joined by `, ` → spliced into `f(input)`;
  - `output` = source segment of the other side of the comparison.
  - Non-`==` forms (keyword args, `abs(...) < eps` float tolerance, `in`,
    boolean asserts) are skipped.
- **id** = `mbpp_<task_id>_<assert index>` (index runs globally over all of a
  problem's assertions).

## Usage

```bash
python convert.py   # download and save_to_disk into ./data
```

```python
from dataset.sources import load_rows
from cwm.training.data import build_dataset

rows = load_rows(["mbpp"])                                   # or ["mbpp", "humaneval"]
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
