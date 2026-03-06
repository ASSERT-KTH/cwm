# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
CRUXEval-O evaluation for CWM.

Uses the canonical direct-output prompt format from the original CRUXEval benchmark
and CWM's FastGen/ImpGen inference stack.

Example (2 GPUs, 1 TP group):

    python -m torch.distributed.run --nproc_per_node=2 \\
        -m evals.cruxeval.run_eval \\
        checkpoint_dir=/path/to/cwm \\
        dump_dir=./eval-cwm-cruxeval

Example (8 GPUs, 4 TP groups of 2 → 4-way data parallelism):

    python -m torch.distributed.run --nproc_per_node=8 \\
        -m evals.cruxeval.run_eval \\
        checkpoint_dir=/path/to/cwm \\
        gen_args.tp_size=2 \\
        dump_dir=./eval-cwm-cruxeval
"""

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
from evals.cruxeval.evaluate import (
    check_correct,
    extract_answer,
    extract_answer_reasoning,
    extract_answer_trace_full,
    extract_answer_trace_single_step,
)
from evals.cruxeval.prompts import (
    make_direct_output_prompt,
    make_reasoning_prompt_tokens,
    make_trace_full_prompt_tokens,
    make_trace_single_step_prompt_tokens,
)

logger = logging.getLogger(__name__)


_MAX_GEN: dict[str, int] = {
    "direct": 512,
    "reasoning": 8192,
    "trace_full": 8192,
    "trace_single_step": 512,
}


@dataclass
class CruxEvalArgs:
    checkpoint_dir: str = ""
    dump_dir: str = "eval-cwm-cruxeval"
    # Evaluation mode: direct | reasoning | trace_full | trace_single_step
    mode: str = "direct"
    # Number of samples to evaluate; -1 evaluates all 800
    n_samples: int = -1
    # Max tokens to generate per sample (0 = use mode default)
    max_gen: int = 0
    seed: int = 42
    gen_args: FastGenArgs = field(
        default_factory=lambda: FastGenArgs(
            tp_size=2,
            use_sampling=False,
            temperature=0.0,
        )
    )
    setup: SetupArgs = field(default_factory=SetupArgs)


def setup_mesh(args: CruxEvalArgs) -> tuple[DeviceMesh, torch.distributed.ProcessGroup]:
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
        # moodist backend for ImpGen queue compatibility
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
    mode: str,
    exc_queue: queue.Queue,
    done_event: threading.Event,
) -> None:
    """
    Worker thread: only runs on TP rank 0 within each DP group.
    Iterates over samples, calls g.generate() (blocking), extracts answer,
    checks correctness. Sets done_event when finished so the main thread
    can drive g.stop() from its work loop.
    """
    try:
        for sample in samples:
            code = sample["code"]
            inp = sample["input"]

            if mode == "direct":
                prompt_tokens = g.tokenizer.encode(
                    make_direct_output_prompt(code, inp), bos=True
                )
                packet = g.generate(
                    tokens=prompt_tokens,
                    max_gen=max_gen,
                    temperature=0.0,
                    stop_str="[/ANSWER]",
                )
                generation = g.tokenizer.decode(packet.tokens, cut_at_stop_tokens=False)
                predicted = extract_answer(generation, inp)

            elif mode == "reasoning":
                prompt_tokens = make_reasoning_prompt_tokens(code, inp, g.tokenizer)
                packet = g.generate(
                    tokens=prompt_tokens,
                    max_gen=max_gen,
                    temperature=0.0,
                    stop_str="[/ANSWER]",
                )
                generation = g.tokenizer.decode(packet.tokens, cut_at_stop_tokens=False)
                predicted = extract_answer_reasoning(generation, inp)

            elif mode == "trace_full":
                prompt_tokens = make_trace_full_prompt_tokens(code, inp, g.tokenizer)
                packet = g.generate(
                    tokens=prompt_tokens,
                    max_gen=max_gen,
                    temperature=0.0,
                )
                generation = g.tokenizer.decode(packet.tokens, cut_at_stop_tokens=False)
                predicted = extract_answer_trace_full(generation, inp)

            elif mode == "trace_single_step":
                prompt_tokens = make_trace_single_step_prompt_tokens(code, inp, g.tokenizer)
                packet = g.generate(
                    tokens=prompt_tokens,
                    max_gen=max_gen,
                    temperature=0.0,
                    stop_str="<|frame_sep|>",
                )
                generation = g.tokenizer.decode(packet.tokens, cut_at_stop_tokens=False)
                predicted = extract_answer_trace_single_step(generation, inp)

            else:
                raise ValueError(f"Unknown mode: {mode!r}")

            correct = (
                check_correct(code, sample["output"], predicted)
                if predicted is not None
                else False
            )

            results.append(
                {
                    "id": sample["id"],
                    "mode": mode,
                    "generation": generation,
                    "predicted": predicted,
                    "expected": sample["output"],
                    "correct": correct,
                }
            )
            pbar.update(1)
    except Exception as e:
        exc_queue.put(e)
        logger.exception("Exception in eval worker")
    finally:
        # Signal the main thread to start calling g.stop(). Do NOT call
        # g.stop() here: ImpGen.stop() needs to be called repeatedly from
        # the main work loop on all TP ranks so both ranks put into
        # _stop_queue before rank 0 sends the None termination packet.
        done_event.set()


def main(args: CruxEvalArgs) -> None:
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

    # Load dataset and partition across DP ranks
    dataset = list(load_dataset("cruxeval-org/cruxeval", split="test"))
    if args.n_samples > 0:
        dataset = dataset[: args.n_samples]
    my_samples = dataset[dp_rank::n_dp]
    logger.info(f"DP rank {dp_rank}/{n_dp}: evaluating {len(my_samples)} samples")

    dump_path = Path(args.dump_dir)
    if is_rank_zero:
        dump_path.mkdir(parents=True, exist_ok=True)

    # Only TP rank 0 within each DP group submits generation requests.
    # TP rank 1 only drives g.work() — it must never call g.generate()
    # because ImpGen routes futures through _tp_queue which only rank 0
    # dequeues via _sync_q, so rank 1's futures would never be resolved.
    is_tp_rank_zero = tp_group.rank() == 0

    results: list[dict] = []
    exc_queue: queue.Queue = queue.Queue()
    done_event = threading.Event()

    effective_max_gen = args.max_gen if args.max_gen != 0 else _MAX_GEN[args.mode]

    if is_tp_rank_zero:
        pbar = tqdm(total=len(my_samples), desc=f"CRUXEval-O [dp={dp_rank}]", position=dp_rank)
        worker = threading.Thread(
            target=run_eval_worker,
            args=(
                my_samples,
                g,
                results,
                pbar,
                effective_max_gen,
                args.mode,
                exc_queue,
                done_event,
            ),
            daemon=True,
        )
        worker.start()
    else:
        # Non-worker ranks signal done immediately so the work loop below
        # starts calling g.stop() right away, allowing the stop handshake
        # across both TP ranks to complete.
        done_event.set()

    # Drive FastGen on the main thread. Once the worker is done, call
    # g.stop() on every iteration — ImpGen requires repeated calls so that
    # both TP ranks contribute to _stop_queue before rank 0 sends the None
    # termination packet to unblock the FastGen generator.
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

    # Only TP rank 0 has results; TP rank 1 skips writing
    if is_tp_rank_zero:
        rank_results_path = dump_path / f"results_dp{dp_rank}.jsonl"
        with rank_results_path.open("w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

        n_correct = sum(r["correct"] for r in results)
        logger.info(
            f"DP rank {dp_rank}: {n_correct}/{len(results)} correct "
            f"({100 * n_correct / len(results):.1f}%)"
        )

    torch.distributed.barrier()

    # Rank 0 aggregates all per-rank files
    if is_rank_zero:
        all_results: list[dict] = []
        for rank in range(n_dp):
            rank_path = dump_path / f"results_dp{rank}.jsonl"
            with rank_path.open("r") as f:
                all_results.extend(json.loads(line) for line in f)

        # Sort by original sample order to make output deterministic
        id_to_idx = {s["id"]: i for i, s in enumerate(dataset)}
        all_results.sort(key=lambda r: id_to_idx.get(r["id"], 0))

        total_correct = sum(r["correct"] for r in all_results)
        pass_at_1 = total_correct / len(all_results)
        print(f"\nCRUXEval-O pass@1: {pass_at_1:.4f} ({total_correct}/{len(all_results)})")

        with (dump_path / "results.jsonl").open("w") as f:
            for r in all_results:
                f.write(json.dumps(r) + "\n")

        summary = {
            "pass_at_1": pass_at_1,
            "n_correct": total_correct,
            "n_total": len(all_results),
        }
        with (dump_path / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Results written to {dump_path}")

    fg.destroy()

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = load_from_cli(CruxEvalArgs)
    main(args)
