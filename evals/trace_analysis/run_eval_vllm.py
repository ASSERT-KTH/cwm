# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
vLLM backend for the Table 9 eval -- true 4-bit AWQ (compressed-tensors) at low
VRAM. Same prompt, scoring, and dump schema as run_eval.py / run_eval_hf.py
(score.py re-scores these runs too). Greedy decoding.

    python -m evals.trace_analysis.run_eval_vllm \\
        --model model_weights/cwm-awq-4bit --dump_dir eval-cwm-table9-awq-4bit
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from evals.cruxeval.evaluate import check_correct, extract_answer_trace_full
from dataset.cruxeval.ground_truth import ground_truth_trace
from evals.trace_analysis.metrics import compute_trace_metrics, format_table9
from evals.trace_analysis.run_eval_hf import (
    _aggregate_results,
    build_trace_full_prompt_ids,
)
from dataset.cruxeval.trace_format import parse_generated_trace

logger = logging.getLogger(__name__)

_MAX_GEN = 8192


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model_weights/cwm-awq-4bit")
    parser.add_argument("--dump_dir", default="eval-cwm-table9-awq-4bit")
    parser.add_argument("--n_samples", type=int, default=-1)
    parser.add_argument("--max_gen", type=int, default=_MAX_GEN)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--gpu_mem_util", type=float, default=0.9)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    # quantization (compressed-tensors AWQ) is auto-detected from the checkpoint.
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=args.tp_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
    )

    dataset = list(load_dataset("cruxeval-org/cruxeval", split="test"))
    if args.n_samples > 0:
        dataset = dataset[: args.n_samples]

    prompts = [
        {"prompt_token_ids": build_trace_full_prompt_ids(s["code"], s["input"], tok)}
        for s in dataset
    ]
    # Greedy decoding (Table 9).
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_gen)
    outputs = llm.generate(prompts, sp)  # vLLM preserves input order

    def token_len(s: str) -> int:
        return len(tok.encode(s, add_special_tokens=False))

    results: list[dict] = []
    for sample, out in zip(dataset, outputs):
        code = sample["code"]
        inp = sample["input"]
        expected = sample["output"]

        gen_ids = list(out.outputs[0].token_ids)
        generation = tok.decode(gen_ids, skip_special_tokens=False)

        predicted = extract_answer_trace_full(generation, inp)
        correct = (
            check_correct(code, expected, predicted)
            if predicted is not None
            else False
        )

        pred_frames, well_formed = parse_generated_trace(generation)
        gt_frames, gt_error = ground_truth_trace(code, inp)
        m = compute_trace_metrics(
            gt_frames, pred_frames, well_formed, token_len=token_len
        )

        results.append(
            {
                "id": sample["id"],
                "code": code,
                "input": inp,
                "expected": expected,
                "predicted": predicted,
                "correct": correct,
                "gt_error": gt_error,
                "n_gt_frames": len(gt_frames),
                "n_pred_frames": len(pred_frames),
                "metrics": dataclasses.asdict(m),
                "generation": generation,
            }
        )

    dump_path = Path(args.dump_dir)
    dump_path.mkdir(parents=True, exist_ok=True)

    agg = _aggregate_results(results)
    print("\n" + format_table9(agg, column_label="CruxEval") + "\n")

    with (dump_path / "results.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    summary = {
        "table9": dataclasses.asdict(agg),
        "n_total": len(results),
        "decoding": "greedy",
        "model": args.model,
        "loader": "vllm",
    }
    with (dump_path / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Results written to %s", dump_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
