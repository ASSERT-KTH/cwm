"""Extract decode-step activation trajectories for bug-fixing experiments.

Processes (original, buggy) pairs from interp/bug_trace/data/pairs.json.
For each sample, captures the hidden state at the last token position at
every decode step (h_T[-1, L]), giving a representation trajectory over time.

Supports two tracks:
  track_a: NL reasoning mode — model reasons in <think> and outputs the fix
  track_b: trace_full mode — model traces the (buggy) code execution

Example (8 GPUs, TP=2, DP=4):

    N_GPUS=8 python -m torch.distributed.run --nproc_per_node=8 \\
        -m interp.bug_trace.run_bug_extract \\
        checkpoint_dir=./model_weights/cwm \\
        dump_dir=./interp-bug-trajectories \\
        pairs_path=./interp/bug_trace/data/pairs.json \\
        track=track_a \\
        gen_args.tp_size=2 \\
        gen_args.num_cuda_graphs=0
"""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.distributed
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
from interp.bug_trace.dataset import BugSample, load_pairs, pairs_to_samples
from interp.bug_trace.prompts import make_bug_fix_prompt_tokens, make_bug_trace_prompt_tokens
from interp.extract.hooks import (
    ActivationStore,
    activation_hook_context,
    install_forward_hooks,
    uninstall_forward_hooks,
)

logger = logging.getLogger(__name__)

_MAX_GEN = {
    "track_a": 4096,
    "track_b": 8192,
}

# Layers to capture (balance between resolution and memory)
_DEFAULT_LAYERS = [16, 32, 48, 63]

# Capture every K-th decode step (memory control)
_STRIDE = 5


@dataclass
class BugExtractArgs:
    checkpoint_dir: str = "./model_weights/cwm"
    dump_dir: str = "interp-bug-trajectories"
    pairs_path: str = "interp/bug_trace/data/pairs.json"
    track: str = "track_a"        # track_a | track_b
    layers: list[int] = field(default_factory=lambda: list(_DEFAULT_LAYERS))
    stride: int = _STRIDE         # capture every N-th decode step
    n_samples: int = 0            # 0 = all pairs
    include_originals: bool = True  # if False, only extract buggy samples (avoids contradictory prompts)
    max_gen: int = 0              # 0 = use track default (_MAX_GEN); set explicitly to override
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


def setup_mesh(args: BugExtractArgs) -> tuple[DeviceMesh, torch.distributed.ProcessGroup]:
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


def _extract_bugfix_answer(generated_text: str) -> str | None:
    """Extract the fixed code from Track-A generation.

    Looks for text after </think> and tries to extract a Python function.
    """
    think_end = "</think>"
    idx = generated_text.find(think_end)
    if idx == -1:
        return None
    after = generated_text[idx + len(think_end):].strip()
    # Strip code fences if present
    if after.startswith("```python"):
        after = after[9:]
    elif after.startswith("```"):
        after = after[3:]
    if "```" in after:
        after = after[: after.index("```")]
    return after.strip() if after.strip() else None


def _check_bugfix_correct(code: str, fixed_code: str | None, input_str: str, correct_output: str) -> bool:
    """Check if the fixed code produces the correct output."""
    if fixed_code is None:
        return False
    try:
        globs: dict = {}
        exec(compile(fixed_code, "<string>", "exec"), globs)  # noqa: S102
        f = globs.get("f")
        if f is None:
            return False
        result = eval(f"f({input_str})", globs)  # noqa: S307
        return repr(result).strip() == correct_output.strip()
    except Exception:
        return False


def _build_prompt_and_extract(
    sample: BugSample,
    tokenizer,
    track: str,
) -> tuple[list[int], callable, str | None]:
    """Return (prompt_tokens, extract_fn, stop_str)."""
    if track == "track_a":
        tokens = make_bug_fix_prompt_tokens(
            buggy_code=sample.code,
            input_str=sample.input_str,
            wrong_output=sample.wrong_output,
            correct_output=sample.correct_output,
            tokenizer=tokenizer,
        )
        def extract_fn(gen):
            fixed = _extract_bugfix_answer(gen)
            correct = _check_bugfix_correct(
                sample.code, fixed, sample.input_str, sample.correct_output
            )
            return fixed, correct
        stop_str = None
    elif track == "track_b":
        tokens = make_bug_trace_prompt_tokens(
            buggy_code=sample.code,
            input_str=sample.input_str,
            tokenizer=tokenizer,
        )
        def extract_fn(gen):
            predicted = extract_answer_trace_full(gen, sample.input_str)
            correct = (
                check_correct(sample.code, sample.correct_output, predicted)
                if predicted is not None
                else False
            )
            return predicted, correct
        stop_str = None
    else:
        raise ValueError(f"Unknown track: {track!r}")
    return tokens, extract_fn, stop_str


def run_extract_worker(
    samples: list[BugSample],
    g: ImpGen,
    results: list[dict],
    pbar: tqdm,
    max_gen: int,
    track: str,
    layers: list[int],
    stride: int,
    dump_dir: Path,
    exc_queue: queue.Queue,
    done_event: threading.Event,
) -> None:
    try:
        for sample in samples:
            act_path = dump_dir / "trajectories" / f"{sample.sample_id}.pt"
            if act_path.exists():
                try:
                    existing = torch.load(act_path, map_location="cpu", weights_only=False)
                    results.append({k: v for k, v in existing.items() if k != "trajectory"})
                except Exception as e:
                    logger.warning(f"Could not reload {act_path}: {e}")
                pbar.update(1)
                continue

            prompt_tokens, extract_fn, stop_str = _build_prompt_and_extract(
                sample, g.tokenizer, track
            )
            n_prompt = len(prompt_tokens)

            # Capture ALL positions (decode trajectory extracted post-hoc)
            store = ActivationStore(layers=layers, capture_token_ids=None)

            gen_kwargs: dict = dict(max_gen=max_gen)
            if stop_str:
                gen_kwargs["stop_str"] = stop_str

            with activation_hook_context(store=store):
                packet = g.generate(tokens=prompt_tokens, **gen_kwargs)

            generated_text = g.tokenizer.decode(packet.tokens, cut_at_stop_tokens=False)
            predicted, correct = extract_fn(generated_text)

            # Extract decode trajectory: rows [n_prompt:] in captured activations
            # store.get_activations(L) returns [n_prompt + n_decode_steps, dim]:
            #   - First n_prompt rows = prefill token activations (one per prompt token)
            #   - Remaining rows = decode step activations (one per generated token)
            # NOTE: store.positions is shared across all captured layers and accumulates
            # n_layers entries per forward pass — do NOT use it for per-layer indexing.
            trajectory: dict[int, torch.Tensor] = {}
            for layer in layers:
                full = store.get_activations(layer)
                if full is None:
                    continue
                # Slice off the prefill rows to get decode-only activations
                decode_acts = full[n_prompt:]  # [n_decode_steps, dim]
                if decode_acts.shape[0] == 0:
                    continue

                # Stride subsampling
                if stride > 1:
                    decode_acts = decode_acts[::stride]

                trajectory[layer] = decode_acts.half()  # save in fp16 for space

            # Corresponding decode token IDs (strided)
            generated_ids = packet.tokens  # all generated tokens (not including prompt)
            decode_token_ids = list(generated_ids[::stride]) if stride > 1 else list(generated_ids)

            sample_data = {
                "sample_id": sample.sample_id,
                "pair_id": sample.pair_id,
                "original_id": sample.original_id,
                "is_buggy": sample.is_buggy,
                "mutation_type": sample.mutation_type,
                "track": track,
                "code": sample.code,
                "input_str": sample.input_str,
                "correct_output": sample.correct_output,
                "wrong_output": sample.wrong_output,
                "generated_text": generated_text,
                "extracted_answer": predicted,
                "correct": correct,
                "n_prompt_tokens": n_prompt,
                "n_generated_tokens": len(packet.tokens),
                "decode_token_ids": decode_token_ids,
                "stride": stride,
                "trajectory": trajectory,  # {layer: [T/stride, dim]}
            }

            torch.save(sample_data, act_path)

            index_entry = {k: v for k, v in sample_data.items() if k != "trajectory"}
            results.append(index_entry)
            pbar.update(1)

    except Exception as e:
        exc_queue.put(e)
        logger.exception("Exception in extract worker")
    finally:
        done_event.set()


def _save_run_config(dump_path: Path, args: BugExtractArgs) -> None:
    """Save run config for reproducibility."""
    import datetime
    config = {
        "args": dataclass_to_dict(args),
        "git_commit": _get_git_commit(),
        "date": datetime.datetime.now().isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "torch_version": torch.__version__,
    }
    with (dump_path / "run_config.json").open("w") as f:
        json.dump(config, f, indent=2)


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def main(args: BugExtractArgs) -> None:
    setup_env(mp_spawn_method=args.setup.spawn_method)
    init_torch_distributed(timeout=args.setup.torch_init_timeout)
    setup_torch_flags(**dataclass_to_dict(args.setup))
    set_seed(args.seed)

    world_mesh, tp_group = setup_mesh(args)
    dp_rank: int = world_mesh["dp"].get_local_rank()
    n_dp: int = world_mesh["dp"].size()
    is_rank_zero = get_is_rank_zero()
    is_tp_rank_zero = tp_group.rank() == 0

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

        # Load dataset
        pairs = load_pairs(args.pairs_path)
        all_samples = pairs_to_samples(pairs, include_originals=args.include_originals)
        if args.n_samples > 0:
            all_samples = all_samples[: args.n_samples]

        # DP sharding
        my_samples = all_samples[dp_rank::n_dp]
        logger.info(f"DP rank {dp_rank}/{n_dp}: processing {len(my_samples)} samples")

        dump_path = Path(args.dump_dir)
        traj_dir = dump_path / "trajectories"
        if is_rank_zero:
            dump_path.mkdir(parents=True, exist_ok=True)
            _save_run_config(dump_path, args)
        if is_tp_rank_zero:
            traj_dir.mkdir(parents=True, exist_ok=True)
        torch.distributed.barrier()

        results: list[dict] = []
        exc_queue: queue.Queue = queue.Queue()
        done_event = threading.Event()
        max_gen = args.max_gen if args.max_gen > 0 else _MAX_GEN.get(args.track, 4096)

        if is_tp_rank_zero:
            pbar = tqdm(
                total=len(my_samples),
                desc=f"BugExtract [{args.track}, dp={dp_rank}]",
                position=dp_rank,
            )
            worker = threading.Thread(
                target=run_extract_worker,
                args=(
                    my_samples, g, results, pbar, max_gen,
                    args.track, args.layers, args.stride,
                    dump_path, exc_queue, done_event,
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
                raise RuntimeError("Exception in worker") from exc
            except queue.Empty:
                pass

        if is_tp_rank_zero:
            worker.join()
            pbar.close()

        if not exc_queue.empty():
            raise RuntimeError("Exception in worker") from exc_queue.get()

        # Write per-rank index
        if is_tp_rank_zero:
            rank_idx = dump_path / f"index_dp{dp_rank}.jsonl"
            with rank_idx.open("w") as f:
                for entry in results:
                    f.write(json.dumps(entry, default=str) + "\n")

        torch.distributed.barrier()

        # Aggregate on rank 0
        if is_rank_zero:
            all_entries: list[dict] = []
            for rank in range(n_dp):
                rp = dump_path / f"index_dp{rank}.jsonl"
                if rp.exists():
                    with rp.open() as f:
                        all_entries.extend(json.loads(l) for l in f)

            all_entries.sort(key=lambda e: e.get("sample_id", ""))

            with (dump_path / "index.jsonl").open("w") as f:
                for entry in all_entries:
                    f.write(json.dumps(entry, default=str) + "\n")

            n_correct = sum(e.get("correct", False) for e in all_entries)
            n_buggy = sum(e.get("is_buggy", False) for e in all_entries)
            print(
                f"\nExtraction complete: {len(all_entries)} samples "
                f"({n_buggy} buggy), pass@1={n_correct/max(len(all_entries), 1):.4f}"
            )

            summary = {
                "n_samples": len(all_entries),
                "n_buggy": n_buggy,
                "n_original": len(all_entries) - n_buggy,
                "pass_at_1": n_correct / max(len(all_entries), 1),
                "track": args.track,
                "layers": args.layers,
                "stride": args.stride,
            }
            with (dump_path / "summary.json").open("w") as f:
                json.dump(summary, f, indent=2)

            try:
                import wandb
                if args.wandb_project:
                    wandb.init(
                        project=args.wandb_project,
                        name=args.wandb_run_name or f"bug-extract-{args.track}",
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
    args = load_from_cli(BugExtractArgs)
    main(args)
