# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Reproduction of Table 9 from "CWM: An Open-Weights LLM for Research on Code
Generation with World Models" (arXiv:2510.02387): a detailed, component-wise
analysis of CWM's full execution-trace prediction on CRUXEval.

For each CRUXEval sample we prompt CWM in full-trace mode with **greedy
decoding** (as in the paper), parse the generated trace, build a ground-truth
trace by executing the function under ``sys.settrace`` in CWM's trace format,
and score the individual components:

    Output     pass@1
    Trace      Valid Trace Format / State Exact Match / Action Exact Match
    States     Valid JSON Format / Key Match / Key+Value Match
    Statistics Avg State Length (Token) / Avg Action Length (Token)

Only the CRUXEval column is reproducible from open data; the paper's
"Function-level" column uses Meta-internal data. See README.md for the known
gaps between this re-implementation and the paper's internal tracer.

Example (8 GPUs, 4 TP groups of 2 -> 4-way data parallelism):

    python -m torch.distributed.run --nproc_per_node=8 \\
        -m evals.trace_analysis.run_eval \\
        checkpoint_dir=/path/to/cwm \\
        gen_args.tp_size=2 \\
        dump_dir=./eval-cwm-table9
"""

import dataclasses
import json
import logging
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.distributed
from datasets import load_dataset
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from tqdm import tqdm

from cwm.common.environment import (
    get_is_rank_zero,
    get_world_size,
    init_torch_distributed,
    set_seed,
    setup_env,
    setup_torch_flags,
)
from cwm.common.params import dataclass_to_dict, load_from_cli
from cwm.fastgen.generate import FastGen
from cwm.fastgen.utils.loading import build_fastgen_model, build_tokenizer_from_ckpt
from cwm.rl.lib.impgen import ImpGen
from evals.args import FastGenArgs, SetupArgs
from evals.cruxeval.evaluate import check_correct, extract_answer_trace_full
from evals.cruxeval.prompts import make_trace_full_prompt_tokens
from evals.trace_analysis.ground_truth import ground_truth_trace
from evals.trace_analysis.metrics import (
    Table9Aggregate,
    aggregate,
    compute_trace_metrics,
    format_table9,
)
from evals.trace_analysis.trace_format import parse_generated_trace

logger = logging.getLogger(__name__)

# Full traces can be long; 8192 matches the cruxeval trace_full budget.
_MAX_GEN = 8192


@dataclass
class TraceAnalysisArgs:
    checkpoint_dir: str = ""
    dump_dir: str = "eval-cwm-table9"
    # Number of samples to evaluate; -1 evaluates all 800.
    n_samples: int = -1
    max_gen: int = 0  # 0 -> use _MAX_GEN
    seed: int = 42
    # Table 9 uses greedy decoding: use_sampling defaults to False here.
    gen_args: FastGenArgs = field(
        default_factory=lambda: FastGenArgs(
            tp_size=2,
            use_sampling=False,
            temperature=1.0,
            top_p=1.0,
        )
    )
    setup: SetupArgs = field(
        default_factory=lambda: SetupArgs(torch_init_timeout=7200)
    )


def setup_mesh(
    args: TraceAnalysisArgs,
) -> tuple[DeviceMesh, torch.distributed.ProcessGroup]:
    world_size = get_world_size()
    tp_size = args.gen_args.tp_size
    num_tp_groups = world_size // tp_size
    assert num_tp_groups * tp_size == world_size, (
        f"tp_size must divide world_size: {world_size=} {tp_size=}"
    )
    world_mesh = init_device_mesh(
        device_type="cuda",
        mesh_shape=(num_tp_groups, tp_size),
        mesh_dim_names=("dp", "tp"),
    )
    global_rank = world_mesh.get_rank()
    all_tp_group_ranks = world_mesh.mesh.tolist()
    tp_group = None
    for ranks in all_tp_group_ranks:
        pg = torch.distributed.new_group(ranks, backend="moodist")
        if global_rank in ranks:
            tp_group = pg
    assert tp_group is not None
    return world_mesh, tp_group


def run_eval_worker(
    samples: list[dict],
    g: ImpGen,
    results: list[dict],
    pbar: tqdm,
    max_gen: int,
    use_sampling: bool,
    temperature: float,
    top_p: float,
    exc_queue: queue.Queue,
    done_event: threading.Event,
) -> None:
    """Worker thread (TP rank 0 only): generate a full trace per sample, then
    score it against the ground-truth trace."""

    def token_len(s: str) -> int:
        return len(g.tokenizer.encode(s, bos=False))

    try:
        for sample in samples:
            code = sample["code"]
            inp = sample["input"]
            expected = sample["output"]

            prompt_tokens = make_trace_full_prompt_tokens(code, inp, g.tokenizer)
            gen_kwargs = dict(max_gen=max_gen)
            if use_sampling:
                gen_kwargs.update(temperature=temperature, top_p=top_p)
            packet = g.generate(tokens=prompt_tokens, **gen_kwargs)
            generation = g.tokenizer.decode(packet.tokens, cut_at_stop_tokens=False)

            # Output pass@1 via the CRUXEval answer extractor + execution.
            predicted = extract_answer_trace_full(generation, inp)
            correct = (
                check_correct(code, expected, predicted)
                if predicted is not None
                else False
            )

            # Component breakdown: parse predicted trace, build ground truth,
            # and align frame-by-frame.
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
                    "metrics": {
                        "valid_trace_format": m.valid_trace_format,
                        "state_exact_match": m.state_exact_match,
                        "action_exact_match": m.action_exact_match,
                        "valid_json_format": m.valid_json_format,
                        "key_match": m.key_match,
                        "key_value_match": m.key_value_match,
                        "avg_state_tokens": m.avg_state_tokens,
                        "avg_action_tokens": m.avg_action_tokens,
                    },
                    "generation": generation,
                }
            )
            pbar.update(1)
    except Exception as e:
        exc_queue.put(e)
        logger.exception("Exception in eval worker")
    finally:
        done_event.set()


def main(args: TraceAnalysisArgs) -> None:
    setup_env(mp_spawn_method=args.setup.spawn_method)
    init_torch_distributed(timeout=args.setup.torch_init_timeout)
    setup_torch_flags(**dataclass_to_dict(args.setup))
    set_seed(args.seed)

    world_mesh, tp_group = setup_mesh(args)

    dp_rank: int = world_mesh["dp"].get_local_rank()
    n_dp: int = world_mesh["dp"].size()
    is_rank_zero = get_is_rank_zero()

    tokenizer = build_tokenizer_from_ckpt(args.checkpoint_dir)
    model = build_fastgen_model(
        world_mesh=world_mesh,
        checkpoint_dir=args.checkpoint_dir,
        vocab_parallel=args.gen_args.vocab_parallel,
        loss_parallel=args.gen_args.loss_parallel,
    )
    fg = FastGen(
        args.gen_args,
        model=model,
        tokenizer=tokenizer,
        dtype=torch.bfloat16,
        device=torch.device(f"cuda:{torch.cuda.current_device()}"),
        tp_mesh=world_mesh["tp"],
    )
    torch.cuda.empty_cache()
    g = ImpGen(fg, tp_group.rank(), tp_group)

    dataset = list(load_dataset("cruxeval-org/cruxeval", split="test"))
    if args.n_samples > 0:
        dataset = dataset[: args.n_samples]
    my_samples = dataset[dp_rank::n_dp]
    logger.info(f"DP rank {dp_rank}/{n_dp}: evaluating {len(my_samples)} samples")

    dump_path = Path(args.dump_dir)
    if is_rank_zero:
        dump_path.mkdir(parents=True, exist_ok=True)

    is_tp_rank_zero = tp_group.rank() == 0
    results: list[dict] = []
    exc_queue: queue.Queue = queue.Queue()
    done_event = threading.Event()

    effective_max_gen = args.max_gen if args.max_gen != 0 else _MAX_GEN

    if is_tp_rank_zero:
        pbar = tqdm(
            total=len(my_samples), desc=f"Table9 [dp={dp_rank}]", position=dp_rank
        )
        worker = threading.Thread(
            target=run_eval_worker,
            args=(
                my_samples,
                g,
                results,
                pbar,
                effective_max_gen,
                args.gen_args.use_sampling,
                args.gen_args.temperature,
                args.gen_args.top_p,
                exc_queue,
                done_event,
            ),
            daemon=True,
        )
        worker.start()
    else:
        done_event.set()

    while True:
        done = g.work()
        if done:
            break
        if done_event.is_set():
            g.stop()
        try:
            exc = exc_queue.get_nowait()
            raise RuntimeError("Exception in eval worker") from exc
        except queue.Empty:
            pass

    if is_tp_rank_zero:
        worker.join()
        pbar.close()

    if not exc_queue.empty():
        raise RuntimeError("Exception in eval worker") from exc_queue.get()

    if is_tp_rank_zero:
        rank_results_path = dump_path / f"results_dp{dp_rank}.jsonl"
        with rank_results_path.open("w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

    torch.distributed.barrier()

    if is_rank_zero:
        all_results: list[dict] = []
        for rank in range(n_dp):
            rank_path = dump_path / f"results_dp{rank}.jsonl"
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
            "decoding": "greedy" if not args.gen_args.use_sampling else "sampling",
        }
        with (dump_path / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Results written to {dump_path}")

    fg.destroy()

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def _aggregate_results(all_results: list[dict]) -> Table9Aggregate:
    """Rebuild TraceMetrics from dumped results and macro-average them."""
    from evals.trace_analysis.metrics import TraceMetrics

    per_sample = []
    pass_at_1 = []
    for r in all_results:
        md = r["metrics"]
        per_sample.append(
            TraceMetrics(
                valid_trace_format=md["valid_trace_format"],
                state_exact_match=md["state_exact_match"],
                action_exact_match=md["action_exact_match"],
                valid_json_format=md["valid_json_format"],
                key_match=md["key_match"],
                key_value_match=md["key_value_match"],
                avg_state_tokens=md["avg_state_tokens"],
                avg_action_tokens=md["avg_action_tokens"],
            )
        )
        pass_at_1.append(1.0 if r["correct"] else 0.0)
    return aggregate(per_sample, pass_at_1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = load_from_cli(TraceAnalysisArgs)
    main(args)
