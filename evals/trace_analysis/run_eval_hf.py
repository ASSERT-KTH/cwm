# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
HuggingFace variant of the Table 9 eval (same scoring as ``run_eval.py``).

Loads HF weights via ``AutoModelForCausalLM`` and generates with
``model.generate`` instead of the native FastGen stack. Works for full-precision
``facebook/cwm`` and quantized builds (e.g. AWQ-4bit / compressed-tensors); the
quant config is read from the checkpoint. Prompt and dump schema match the native
eval, so ``score.py`` re-scores these runs too. Greedy decoding.

    torchrun --nproc_per_node=8 -m evals.trace_analysis.run_eval_hf \\
        --model model_weights/cwm-awq-4bit --dump_dir eval-cwm-table9-awq-4bit
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import load_dataset
from tqdm import tqdm

from evals.cruxeval.evaluate import check_correct, extract_answer_trace_full
from evals.cruxeval.prompts import _make_trace_context
from dataset.cruxeval.ground_truth import ground_truth_trace
from evals.trace_analysis.metrics import (
    Table9Aggregate,
    TraceMetrics,
    aggregate,
    compute_trace_metrics,
    format_table9,
)
from dataset.cruxeval.trace_format import parse_generated_trace

logger = logging.getLogger(__name__)

# Full traces can be long; 8192 matches the cruxeval trace_full budget.
_MAX_GEN = 8192


def build_trace_full_prompt_ids(code: str, input_str: str, tok) -> list[int]:
    """HF-tokenizer port of ``make_trace_full_prompt_tokens``.

    Produces the same token sequence as the native eval's seeded full-trace
    prompt: ``[BOS][TRACE_CONTEXT_START]$CONTEXT[FRAME_SEP][CALL_SEP]{}
    [ACTION_SEP]def main():\\n[FRAME_SEP]``.
    """
    sid = tok.convert_tokens_to_ids
    context = _make_trace_context(code, input_str)
    ids = [tok.bos_token_id, sid("<|trace_context_start|>")]
    ids += tok.encode(context, add_special_tokens=False)
    ids += [sid("<|frame_sep|>"), sid("<|call_sep|>")]
    ids += tok.encode("{}", add_special_tokens=False)
    ids += [sid("<|action_sep|>")]
    ids += tok.encode("def main():\n", add_special_tokens=False)
    ids += [sid("<|frame_sep|>")]
    return ids


def _aggregate_results(all_results: list[dict]) -> Table9Aggregate:
    """Rebuild TraceMetrics from dumped results and macro-average them."""
    per_sample = []
    pass_at_1 = []
    for r in all_results:
        md = r["metrics"]
        per_sample.append(TraceMetrics(**md))
        pass_at_1.append(1.0 if r["correct"] else 0.0)
    return aggregate(per_sample, pass_at_1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model_weights/cwm_hf")
    parser.add_argument("--dump_dir", default="eval-cwm-table9-hf")
    parser.add_argument("--n_samples", type=int, default=-1)
    parser.add_argument("--max_gen", type=int, default=_MAX_GEN)
    parser.add_argument(
        "--device_map",
        default="auto",
        help="Only used in single-process mode; ignored under torchrun.",
    )
    args = parser.parse_args()

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    # --- Distributed (data-parallel) setup -------------------------------
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    ddp = world_size > 1
    if ddp:
        # gloo: coordination only (results merge); generation runs on GPU.
        dist.init_process_group(backend="gloo")
    is_rank_zero = rank == 0

    tok = AutoTokenizer.from_pretrained(args.model)
    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (
        eos_id[0] if isinstance(eos_id, list) else eos_id
    )

    load_kwargs: dict = {"dtype": torch.bfloat16, "attn_implementation": "sdpa"}
    if ddp:
        # One full model per GPU. Place at load time via device_map (quantized
        # weights don't reliably support a post-load .to()).
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        load_kwargs["device_map"] = {"": local_rank}
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    else:
        if args.device_map not in (None, "none", "None"):
            load_kwargs["device_map"] = args.device_map
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
        device = model.device
    model.eval()

    def token_len(s: str) -> int:
        return len(tok.encode(s, add_special_tokens=False))

    dataset = list(load_dataset("cruxeval-org/cruxeval", split="test"))
    if args.n_samples > 0:
        dataset = dataset[: args.n_samples]
    my_samples = dataset[rank::world_size] if ddp else dataset
    logger.info("rank %d/%d: %d samples", rank, world_size, len(my_samples))

    dump_path = Path(args.dump_dir)
    if is_rank_zero:
        dump_path.mkdir(parents=True, exist_ok=True)
    if ddp:
        dist.barrier()

    results: list[dict] = []
    pbar = tqdm(total=len(my_samples), desc=f"Table9-HF [rank={rank}]", position=rank)
    for sample in my_samples:
        code = sample["code"]
        inp = sample["input"]
        expected = sample["output"]

        prompt_ids = build_trace_full_prompt_ids(code, inp, tok)
        input_ids = torch.tensor([prompt_ids], device=device)
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=args.max_gen,
                do_sample=False,
                num_beams=1,
                pad_token_id=pad_id,
            )
        gen_ids = out[0][len(prompt_ids):].tolist()
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
        pbar.update(1)
    pbar.close()

    rank_results_path = dump_path / f"results_dp{rank}.jsonl"
    with rank_results_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    if ddp:
        dist.barrier()

    if is_rank_zero:
        all_results: list[dict] = []
        for r in range(world_size):
            rank_path = dump_path / f"results_dp{r}.jsonl"
            with rank_path.open("r") as f:
                all_results.extend(json.loads(line) for line in f)

        id_to_idx = {s["id"]: i for i, s in enumerate(dataset)}
        all_results.sort(key=lambda r: id_to_idx.get(r["id"], 0))

        agg = _aggregate_results(all_results)
        table = format_table9(agg, column_label="CruxEval")
        print("\n" + table + "\n")

        with (dump_path / "results.jsonl").open("w") as f:
            for r in all_results:
                f.write(json.dumps(r) + "\n")

        summary = {
            "table9": dataclasses.asdict(agg),
            "n_total": len(all_results),
            "decoding": "greedy",
            "model": args.model,
            "loader": "huggingface",
        }
        with (dump_path / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Results written to %s", dump_path)

    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
