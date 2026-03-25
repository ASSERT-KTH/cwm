"""Extract activations from CWM during CruxEval inference.

Example (8 GPUs, TP=2, DP=4):

    python -m torch.distributed.run --nproc_per_node=8 \\
        -m interp.extract.run_extract \\
        checkpoint_dir=./model_weights/cwm \\
        dump_dir=./interp-extract-trace_full \\
        mode=trace_full \\
        gen_args.tp_size=2 \\
        gen_args.num_cuda_graphs=0
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
    extract_answer_trace_full,
    extract_answer_trace_single_step,
)
from evals.cruxeval.prompts import (
    make_direct_output_prompt,
    make_trace_full_prompt_tokens,
    make_trace_single_step_prompt_tokens,
)
from interp.extract.hooks import (
    ActivationStore,
    activation_hook_context,
    install_forward_hooks,
    uninstall_forward_hooks,
)

logger = logging.getLogger(__name__)

# Trace separator token IDs — derived at runtime from the tokenizer
# (see _build_trace_token_ids). Placeholder here for backward compat.
_TRACE_TOKEN_IDS: list[int] = []


def _build_trace_token_ids(tokenizer) -> list[int]:
    """Return actual vocab IDs for the 8 trace separator tokens."""
    return [
        tokenizer.frame_sep_id,
        tokenizer.action_sep_id,
        tokenizer.return_sep_id,
        tokenizer.call_sep_id,
        tokenizer.line_sep_id,
        tokenizer.exception_sep_id,
        tokenizer.arg_sep_id,
        tokenizer.trace_context_start_id,
    ]

_MAX_GEN: dict[str, int] = {
    "direct": 512,
    "trace_full": 8192,
    "trace_single_step": 512,
}


@dataclass
class ExtractArgs:
    checkpoint_dir: str = "./model_weights/cwm"
    dump_dir: str = "interp-extract"
    mode: str = "trace_full"
    layers: list[int] = field(
        default_factory=lambda: [0, 8, 16, 24, 32, 40, 48, 56, 63]
    )
    capture_at: str = "trace_tokens"  # trace_tokens | all | last
    n_samples: int = 800
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


def setup_mesh(
    args: ExtractArgs,
) -> tuple[DeviceMesh, torch.distributed.ProcessGroup]:
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
    all_tp_group_ranks = world_mesh.mesh.tolist()
    tp_group = None
    for ranks in all_tp_group_ranks:
        pg = torch.distributed.new_group(ranks, backend="moodist")
        if global_rank in ranks:
            tp_group = pg
    assert tp_group is not None
    return world_mesh, tp_group


def _make_prompt_and_extract_fn(mode: str, sample: dict, g: ImpGen):
    code = sample["code"]
    inp = sample["input"]
    if mode == "direct":
        prompt_tokens = g.tokenizer.encode(
            make_direct_output_prompt(code, inp), bos=True
        )
        extract_fn = lambda gen: extract_answer(gen, inp)
        stop_str = "[/ANSWER]"
    elif mode == "trace_full":
        prompt_tokens = make_trace_full_prompt_tokens(code, inp, g.tokenizer)
        extract_fn = lambda gen: extract_answer_trace_full(gen, inp)
        stop_str = None
    elif mode == "trace_single_step":
        prompt_tokens = make_trace_single_step_prompt_tokens(code, inp, g.tokenizer)
        extract_fn = lambda gen: extract_answer_trace_single_step(gen, inp)
        stop_str = "<|frame_sep|>"
    else:
        raise ValueError(f"Unknown mode: {mode!r}")
    return prompt_tokens, extract_fn, stop_str


def _capture_token_ids_for(capture_at: str, trace_token_ids: list[int]) -> list[int] | None:
    if capture_at == "trace_tokens":
        return trace_token_ids
    elif capture_at == "all":
        return None
    elif capture_at == "last":
        return None  # handled differently during save
    return None


def run_extract_worker(
    samples: list[dict],
    g: ImpGen,
    results: list[dict],
    pbar: tqdm,
    max_gen: int,
    mode: str,
    capture_at: str,
    trace_token_ids: list[int],
    layers: list[int],
    dump_dir: Path,
    exc_queue: queue.Queue,
    done_event: threading.Event,
) -> None:
    capture_ids = _capture_token_ids_for(capture_at, trace_token_ids)

    try:
        for sample in samples:
            act_path = dump_dir / "activations" / f"{sample['id']}.pt"
            if act_path.exists():
                # Rebuild index entry from existing file so index.jsonl stays complete
                try:
                    existing = torch.load(act_path, map_location="cpu", weights_only=False)
                    index_entry = {k: v for k, v in existing.items() if k != "activations"}
                    results.append(index_entry)
                except Exception as e:
                    logger.warning(f"Could not load index for existing {act_path}: {e}")
                pbar.update(1)
                continue

            prompt_tokens, extract_fn, stop_str = _make_prompt_and_extract_fn(
                mode, sample, g
            )

            store = ActivationStore(
                layers=layers,
                capture_token_ids=capture_ids,
            )

            gen_kwargs: dict = dict(max_gen=max_gen)
            if stop_str:
                gen_kwargs["stop_str"] = stop_str

            with activation_hook_context(store=store):
                packet = g.generate(tokens=prompt_tokens, **gen_kwargs)

            generated_text = g.tokenizer.decode(packet.tokens, cut_at_stop_tokens=False)
            predicted = extract_fn(generated_text)
            correct = (
                check_correct(sample["code"], sample["output"], predicted)
                if predicted is not None
                else False
            )

            # Build activations dict: {layer: tensor[n_pos, dim]}
            activations = {}
            for layer in layers:
                act = store.get_activations(layer)
                if act is not None:
                    activations[layer] = act

            sample_data = {
                "sample_id": sample["id"],
                "code": sample["code"],
                "input": sample["input"],
                "output": sample["output"],
                "mode": mode,
                "generated_text": generated_text,
                "extracted_answer": predicted,
                "correct": correct,
                "token_ids": packet.tokens,
                "captured_positions": store.positions,
                "activations": activations,
            }

            torch.save(sample_data, act_path)

            # Index entry (no activations tensor)
            index_entry = {k: v for k, v in sample_data.items() if k != "activations"}
            results.append(index_entry)
            pbar.update(1)

    except Exception as e:
        exc_queue.put(e)
        logger.exception("Exception in extract worker")
    finally:
        done_event.set()


def main(args: ExtractArgs) -> None:
    setup_env(mp_spawn_method=args.setup.spawn_method)
    init_torch_distributed(timeout=args.setup.torch_init_timeout)
    setup_torch_flags(**dataclass_to_dict(args.setup))
    set_seed(args.seed)

    world_mesh, tp_group = setup_mesh(args)
    dp_rank: int = world_mesh["dp"].get_local_rank()
    n_dp: int = world_mesh["dp"].size()
    is_rank_zero = get_is_rank_zero()

    # Install hooks before model inference begins
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
        g = ImpGen(fg, tp_group.rank(), tp_group)

        trace_token_ids = _build_trace_token_ids(tokenizer)
        logger.info(f"Trace token IDs: {trace_token_ids}")

        dataset = list(load_dataset("cruxeval-org/cruxeval", split="test"))
        if args.n_samples > 0:
            dataset = dataset[: args.n_samples]
        my_samples = dataset[dp_rank::n_dp]
        logger.info(f"DP rank {dp_rank}/{n_dp}: extracting {len(my_samples)} samples")

        dump_path = Path(args.dump_dir)
        act_dir = dump_path / "activations"
        if is_rank_zero:
            dump_path.mkdir(parents=True, exist_ok=True)

        # Each DP rank creates its own activations subdir
        if tp_group.rank() == 0:
            act_dir.mkdir(parents=True, exist_ok=True)
        torch.distributed.barrier()

        is_tp_rank_zero = tp_group.rank() == 0
        results: list[dict] = []
        exc_queue: queue.Queue = queue.Queue()
        done_event = threading.Event()

        effective_max_gen = _MAX_GEN.get(args.mode, 4096)

        if is_tp_rank_zero:
            pbar = tqdm(
                total=len(my_samples),
                desc=f"Extract [dp={dp_rank}]",
                position=dp_rank,
            )
            worker = threading.Thread(
                target=run_extract_worker,
                args=(
                    my_samples,
                    g,
                    results,
                    pbar,
                    effective_max_gen,
                    args.mode,
                    args.capture_at,
                    trace_token_ids,
                    args.layers,
                    dump_path,
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
                raise RuntimeError("Exception in extract worker") from exc
            except queue.Empty:
                pass

        if is_tp_rank_zero:
            worker.join()
            pbar.close()

        if not exc_queue.empty():
            raise RuntimeError("Exception in extract worker") from exc_queue.get()

        # Write per-rank index
        if is_tp_rank_zero:
            rank_index_path = dump_path / f"index_dp{dp_rank}.jsonl"
            with rank_index_path.open("w") as f:
                for entry in results:
                    f.write(json.dumps(entry, default=str) + "\n")

        torch.distributed.barrier()

        # Aggregate index on rank 0
        if is_rank_zero:
            all_entries: list[dict] = []
            for rank in range(n_dp):
                rp = dump_path / f"index_dp{rank}.jsonl"
                with rp.open() as f:
                    all_entries.extend(json.loads(line) for line in f)

            id_to_idx = {s["id"]: i for i, s in enumerate(dataset)}
            all_entries.sort(key=lambda e: id_to_idx.get(e["sample_id"], 0))

            with (dump_path / "index.jsonl").open("w") as f:
                for entry in all_entries:
                    f.write(json.dumps(entry, default=str) + "\n")

            n_correct = sum(e["correct"] for e in all_entries)
            pass_at_1 = n_correct / len(all_entries) if all_entries else 0.0
            print(
                f"\nExtraction complete: {len(all_entries)} samples, "
                f"pass@1={pass_at_1:.4f}"
            )

            summary = {
                "n_samples": len(all_entries),
                "pass_at_1": pass_at_1,
                "mode": args.mode,
                "capture_at": args.capture_at,
                "layers": args.layers,
            }
            with (dump_path / "summary.json").open("w") as f:
                import json as _json

                _json.dump(summary, f, indent=2)

            try:
                import wandb

                if args.wandb_run_name or args.wandb_project:
                    wandb.init(
                        project=args.wandb_project,
                        name=args.wandb_run_name or f"extract-{args.mode}",
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
    args = load_from_cli(ExtractArgs)
    main(args)
