"""Run token-level causal interventions on CWM execution traces.

Four intervention types (--intervention):
  variable  : change a variable's value in the trace state JSON, re-generate
  return    : replace the inner function return value, re-generate
  branch    : swap the taken branch for the other branch, re-generate
  corrupt   : inject a foreign sample's trace as the prefix, re-generate

All four use the same two-pass loop:
  pass 1 — generate baseline trace
  pass 2 — apply intervention, generate from modified prefix
  compare baseline vs intervened answers

Example (2 GPUs, TP=2, DP=1):

    python -m torch.distributed.run --nproc_per_node=2 \\
        -m interp.token_intervention.run_intervention \\
        checkpoint_dir=./model_weights/cwm \\
        intervention=variable \\
        varname=n new_value=999 \\
        dump_dir=./interp-intervene-variable \\
        n_samples=50 \\
        gen_args.tp_size=2 \\
        gen_args.num_cuda_graphs=0
"""

import json
import logging
import queue
import random
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
from interp.extract.hooks import install_forward_hooks, uninstall_forward_hooks
from interp.token_intervention.interventions import (
    intervene_branch,
    intervene_corrupt_trace,
    intervene_return,
    intervene_variable,
)

logger = logging.getLogger(__name__)

_MAX_GEN = 8192  # trace_full only


@dataclass
class InterventionArgs:
    checkpoint_dir: str = ""
    dump_dir: str = "interp-intervene"
    intervention: str = "variable"   # variable | return | branch | corrupt
    # --- variable / return params ---
    varname: str = "x"               # variable: which variable to replace
    new_value: str = "0"             # variable / return: replacement value string
    occurrence: int = 0              # variable / branch: which occurrence to target
    which_return: str = "inner"      # return: "inner" or "outer"
    # --- corrupt params ---
    truncate_corrupt: bool = True    # corrupt: truncate foreign trace at inner return
    n_samples: int = 50
    seed: int = 42
    wandb_project: str = "cwm-interp"
    wandb_run_name: str = ""
    gen_args: FastGenArgs = field(
        default_factory=lambda: FastGenArgs(
            tp_size=2,
            use_sampling=False,
            temperature=0.0,
            num_cuda_graphs=0,
        )
    )
    setup: SetupArgs = field(default_factory=lambda: SetupArgs(torch_init_timeout=7200))


def setup_mesh(args: InterventionArgs) -> tuple[DeviceMesh, torch.distributed.ProcessGroup]:
    world_size = get_world_size()
    tp_size = args.gen_args.tp_size
    num_tp_groups = world_size // tp_size
    assert num_tp_groups * tp_size == world_size
    world_mesh = init_device_mesh(
        device_type="cuda",
        mesh_shape=(num_tp_groups, tp_size),
        mesh_dim_names=("dp", "tp"),
    )
    global_rank = world_mesh.get_rank()
    tp_group = None
    for ranks in world_mesh.mesh.tolist():
        pg = torch.distributed.new_group(ranks, backend="moodist")
        if global_rank in ranks:
            tp_group = pg
    assert tp_group is not None
    return world_mesh, tp_group


def _prepare_samples(samples: list[dict], g: ImpGen) -> list[dict]:
    """Pre-tokenise prompts. For 'corrupt', the corrupt_prompt_tokens are also built."""
    prepared = []
    for sample in samples:
        prompt_tokens = make_trace_full_prompt_tokens(
            sample["code"], sample["input"], g.tokenizer
        )
        extract_fn = lambda gen, inp=sample["input"]: extract_answer_trace_full(gen, inp)
        prepared.append(
            {
                "sample": sample,
                "prompt_tokens": prompt_tokens,
                "extract_fn": extract_fn,
            }
        )
    return prepared


def _apply_intervention(
    item: dict,
    baseline_gen: str,
    all_prepared: list[dict],  # needed for 'corrupt' pairing
    idx: int,
    args: InterventionArgs,
    tokenizer,
) -> tuple[list[int] | None, str]:
    """
    Return (new_prefix_tokens, description) or (None, reason) if not applicable.
    """
    prompt_tokens = item["prompt_tokens"]
    code = item["sample"]["code"]

    if args.intervention == "variable":
        tokens = intervene_variable(
            prompt_tokens, baseline_gen, tokenizer,
            varname=args.varname,
            new_value=args.new_value,
            occurrence=args.occurrence,
        )
        desc = f"{args.varname}→{args.new_value}"

    elif args.intervention == "return":
        tokens = intervene_return(
            prompt_tokens, baseline_gen, tokenizer,
            injected_value=args.new_value,
            which=args.which_return,
        )
        desc = f"return→{args.new_value}"

    elif args.intervention == "branch":
        tokens = intervene_branch(
            prompt_tokens, baseline_gen, tokenizer,
            code=code,
            occurrence=args.occurrence,
        )
        desc = "branch_flip"

    elif args.intervention == "corrupt":
        # Pair sample i with sample (i + n//2) % n across the full prepared list
        n = len(all_prepared)
        partner_idx = (idx + n // 2) % n
        partner_gen = all_prepared[partner_idx].get("baseline_gen")
        if partner_gen is None:
            return None, "partner_not_ready"
        tokens = intervene_corrupt_trace(
            prompt_tokens, partner_gen, tokenizer,
            truncate_at_inner_return=args.truncate_corrupt,
        )
        desc = f"corrupt_from_{all_prepared[partner_idx]['sample']['id']}"

    else:
        raise ValueError(f"Unknown intervention: {args.intervention!r}")

    return tokens, desc


def run_intervention_worker(
    prepared: list[dict],
    g: ImpGen,
    args: InterventionArgs,
    results: list[dict],
    pbar: tqdm,
    exc_queue: queue.Queue,
    done_event: threading.Event,
) -> None:
    try:
        for idx, item in enumerate(prepared):
            sample = item["sample"]
            prompt_tokens = item["prompt_tokens"]
            extract_fn = item["extract_fn"]

            # ── Pass 1: baseline ─────────────────────────────────────────────
            packet = g.generate(tokens=prompt_tokens, max_gen=_MAX_GEN)
            baseline_gen = g.tokenizer.decode(packet.tokens, cut_at_stop_tokens=False)
            baseline_ans = extract_fn(baseline_gen)
            baseline_ok = (
                check_correct(sample["code"], sample["output"], baseline_ans)
                if baseline_ans is not None else False
            )
            # Cache for corrupt experiment pairing
            item["baseline_gen"] = baseline_gen

            # ── Intervention ─────────────────────────────────────────────────
            new_prefix, desc = _apply_intervention(
                item, baseline_gen, prepared, idx, args, g.tokenizer
            )

            if new_prefix is None:
                results.append(
                    {
                        "id": sample["id"],
                        "intervention": args.intervention,
                        "found": False,
                        "reason": desc,
                        "baseline_answer": baseline_ans,
                        "baseline_correct": baseline_ok,
                        "intervened_answer": None,
                        "intervened_correct": False,
                        "answer_changed": False,
                    }
                )
                pbar.update(1)
                continue

            # ── Pass 2: intervened ───────────────────────────────────────────
            packet2 = g.generate(tokens=new_prefix, max_gen=_MAX_GEN)
            intervened_gen = g.tokenizer.decode(packet2.tokens, cut_at_stop_tokens=False)
            intervened_ans = extract_fn(intervened_gen)
            intervened_ok = (
                check_correct(sample["code"], sample["output"], intervened_ans)
                if intervened_ans is not None else False
            )

            results.append(
                {
                    "id": sample["id"],
                    "intervention": args.intervention,
                    "desc": desc,
                    "found": True,
                    "baseline_answer": baseline_ans,
                    "baseline_correct": baseline_ok,
                    "intervened_answer": intervened_ans,
                    "intervened_correct": intervened_ok,
                    "answer_changed": baseline_ans != intervened_ans,
                    "intervened_gen": intervened_gen,
                }
            )
            pbar.update(1)

    except Exception as e:
        exc_queue.put(e)
        logger.exception("Exception in intervention worker")
    finally:
        done_event.set()


def _run_samples(
    prepared: list[dict],
    g: ImpGen,
    args: InterventionArgs,
    dp_rank: int,
) -> list[dict]:
    results: list[dict] = []
    exc_queue: queue.Queue = queue.Queue()
    done_event = threading.Event()
    is_tp_rank_zero = g.tp_rank == 0

    if is_tp_rank_zero:
        pbar = tqdm(
            total=len(prepared),
            desc=f"Intervene [{args.intervention}] [dp={dp_rank}]",
            position=dp_rank,
            leave=False,
        )
        worker = threading.Thread(
            target=run_intervention_worker,
            args=(prepared, g, args, results, pbar, exc_queue, done_event),
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
            raise RuntimeError("Exception in intervention worker") from exc
        except queue.Empty:
            pass

    if is_tp_rank_zero:
        worker.join()
        pbar.close()

    if not exc_queue.empty():
        raise RuntimeError("Exception in intervention worker") from exc_queue.get()

    return results


def main(args: InterventionArgs) -> None:
    setup_env(mp_spawn_method=args.setup.spawn_method)
    init_torch_distributed(timeout=args.setup.torch_init_timeout)
    setup_torch_flags(**dataclass_to_dict(args.setup))
    set_seed(args.seed)

    world_mesh, tp_group = setup_mesh(args)
    dp_rank: int = world_mesh["dp"].get_local_rank()
    n_dp: int = world_mesh["dp"].size()
    is_rank_zero = get_is_rank_zero()

    install_forward_hooks()

    try:
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

        dataset = list(load_dataset("cruxeval-org/cruxeval", split="test"))
        random.seed(args.seed)
        random.shuffle(dataset)
        if args.n_samples > 0:
            dataset = dataset[: args.n_samples]
        my_samples = dataset[dp_rank::n_dp]

        dump_path = Path(args.dump_dir)
        if is_rank_zero:
            dump_path.mkdir(parents=True, exist_ok=True)
        torch.distributed.barrier()

        g_tmp = ImpGen(fg, tp_group.rank(), tp_group)
        if g_tmp.tp_rank == 0:
            logger.info("Pre-tokenising %d samples ...", len(my_samples))
        prepared = _prepare_samples(my_samples, g_tmp)

        g = ImpGen(fg, tp_group.rank(), tp_group)
        all_results = _run_samples(prepared, g, args, dp_rank)

        # Write per-rank results
        if g.tp_rank == 0:
            rank_path = dump_path / f"results_dp{dp_rank}.jsonl"
            with rank_path.open("w") as f:
                for r in all_results:
                    f.write(json.dumps(r) + "\n")

        torch.distributed.barrier()

        if is_rank_zero:
            combined: list[dict] = []
            for rank in range(n_dp):
                rp = dump_path / f"results_dp{rank}.jsonl"
                if rp.exists():
                    with rp.open() as f:
                        combined.extend(json.loads(l) for l in f)

            with (dump_path / "results.jsonl").open("w") as f:
                for r in combined:
                    f.write(json.dumps(r) + "\n")

            found   = [r for r in combined if r["found"]]
            n_total = len(combined)
            n_found = len(found)

            baseline_p1    = sum(r["baseline_correct"]   for r in found) / n_found if n_found else 0
            intervened_p1  = sum(r["intervened_correct"] for r in found) / n_found if n_found else 0
            pct_changed    = sum(r["answer_changed"]      for r in found) / n_found if n_found else 0
            pct_found      = n_found / n_total if n_total else 0

            summary = {
                "intervention": args.intervention,
                "n_total": n_total,
                "n_found": n_found,
                "pct_found": pct_found,
                "baseline_pass_at_1": baseline_p1,
                "intervened_pass_at_1": intervened_p1,
                "pct_answer_changed": pct_changed,
            }

            with (dump_path / "summary.json").open("w") as f:
                json.dump(summary, f, indent=2)

            print(f"\n=== Token intervention: {args.intervention} ===")
            print(f"  Samples:           {n_total}")
            print(f"  Intervention found:{n_found}/{n_total}  ({pct_found:.1%})")
            print(f"  Baseline pass@1:   {baseline_p1:.4f}")
            print(f"  Intervened pass@1: {intervened_p1:.4f}  (Δ={intervened_p1-baseline_p1:+.4f})")
            print(f"  Answer changed:    {pct_changed:.1%}")

            try:
                import wandb
                if args.wandb_project:
                    wandb.init(
                        project=args.wandb_project,
                        name=args.wandb_run_name or f"intervene-{args.intervention}",
                        config=dataclass_to_dict(args),
                    )
                    wandb.log(summary)
                    wandb.finish()
            except ImportError:
                pass

        fg.destroy()

    finally:
        uninstall_forward_hooks()

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = load_from_cli(InterventionArgs)
    main(args)
