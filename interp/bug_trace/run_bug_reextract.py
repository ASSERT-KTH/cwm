"""Re-extract activation trajectories using prefill-based approach.

Root cause of original failure: FastGen uses CUDA graphs for decode steps
(controlled by FG_NO_CUDA_GRAPHS env var, NOT by num_cuda_graphs=0 which only
sets the graph count). CUDA graph replay bypasses Python-level _hooked_forward,
so hooks never fire during decode. Result: all trajectory dicts are empty {}.

Fix: use prefill-based re-extraction.
- For each sample, concatenate prompt_tokens + generated_tokens as a single
  "prompt" and run g.generate(full_tokens, max_gen=1).
- The model runs one large prefill pass covering all generated tokens.
- Prefill ALWAYS uses Python _forward (CUDA graphs only apply to decode).
- Causal attention ensures h[n_prompt + T] during prefill equals what h would
  be during decode step T+1 — the activations are mathematically identical.
- Slice h[n_prompt:] from the captured store → decode trajectory.

Usage (8 GPUs, TP=2, DP=4):

    N_GPUS=8 python -m torch.distributed.run --nproc_per_node=8 \\
        -m interp.bug_trace.run_bug_reextract \\
        traj_dir=./interp-bug-trajectories-track_a \\
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
from interp.bug_trace.dataset import BugSample, load_pairs, pairs_to_samples
from interp.bug_trace.prompts import make_bug_fix_prompt_tokens, make_bug_trace_prompt_tokens
from interp.extract.hooks import (
    ActivationStore,
    activation_hook_context,
    install_forward_hooks,
    uninstall_forward_hooks,
)

logger = logging.getLogger(__name__)

_DEFAULT_LAYERS = [16, 32, 48, 63]
_STRIDE = 5


@dataclass
class BugReExtractArgs:
    traj_dir: str = "interp-bug-trajectories-track_a"
    pairs_path: str = "interp/bug_trace/data/pairs.json"
    track: str = "track_a"
    layers: list[int] = field(default_factory=lambda: list(_DEFAULT_LAYERS))
    stride: int = _STRIDE
    checkpoint_dir: str = "./model_weights/cwm"
    seed: int = 42
    gen_args: FastGenArgs = field(
        default_factory=lambda: FastGenArgs(
            tp_size=2,
            use_sampling=False,
            temperature=0.0,
            num_cuda_graphs=0,
        )
    )
    setup: SetupArgs = field(default_factory=lambda: SetupArgs(torch_init_timeout=7200))


def setup_mesh(args: BugReExtractArgs) -> tuple[DeviceMesh, torch.distributed.ProcessGroup]:
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


def _build_prompt_tokens(
    sample: BugSample,
    tokenizer,
    track: str,
) -> list[int]:
    if track == "track_a":
        return make_bug_fix_prompt_tokens(
            buggy_code=sample.code,
            input_str=sample.input_str,
            wrong_output=sample.wrong_output,
            correct_output=sample.correct_output,
            tokenizer=tokenizer,
        )
    elif track == "track_b":
        return make_bug_trace_prompt_tokens(
            buggy_code=sample.code,
            input_str=sample.input_str,
            tokenizer=tokenizer,
        )
    else:
        raise ValueError(f"Unknown track: {track!r}")


def run_reextract_worker(
    samples: list[BugSample],
    g: ImpGen,
    traj_dir: Path,
    layers: list[int],
    stride: int,
    track: str,
    exc_queue: queue.Queue,
    done_event: threading.Event,
    n_skipped: list[int],
    n_updated: list[int],
) -> None:
    try:
        pbar = tqdm(total=len(samples), desc="ReExtract", position=0)

        for sample in samples:
            act_path = traj_dir / "trajectories" / f"{sample.sample_id}.pt"
            if not act_path.exists():
                logger.warning(f"Trajectory file not found: {act_path}")
                pbar.update(1)
                continue

            # Load existing .pt file
            existing = torch.load(act_path, map_location="cpu", weights_only=False)
            existing_traj = existing.get("trajectory", {})

            # Skip if trajectory already populated
            if existing_traj:
                n_skipped[0] += 1
                pbar.update(1)
                continue

            generated_text = existing.get("generated_text", "")
            n_generated = existing.get("n_generated_tokens", 0)

            if not generated_text or n_generated == 0:
                logger.warning(f"Sample {sample.sample_id}: empty generated_text, skipping")
                pbar.update(1)
                continue

            # Rebuild prompt tokens
            prompt_tokens = _build_prompt_tokens(sample, g.tokenizer, track)
            n_prompt = len(prompt_tokens)

            # Tokenize generated text to reconstruct full token sequence.
            # Round-trip encode/decode gives exact same token IDs for CWM tokenizer.
            generated_tokens = g.tokenizer.encode(generated_text, bos=False, eos=False)

            if len(generated_tokens) != n_generated:
                logger.warning(
                    f"Sample {sample.sample_id}: token count mismatch "
                    f"(expected {n_generated}, got {len(generated_tokens)}). "
                    f"Using re-tokenized length."
                )

            if len(generated_tokens) == 0:
                logger.warning(f"Sample {sample.sample_id}: re-tokenization gave 0 tokens, skipping")
                pbar.update(1)
                continue

            # Build full sequence: prompt + generated tokens.
            # Running g.generate on this triggers a prefill over ALL tokens,
            # capturing h at every position (Python path, no CUDA graphs).
            full_tokens = prompt_tokens + generated_tokens

            # Capture all positions (prefill processes ALL tokens in full_tokens)
            store = ActivationStore(layers=layers, capture_token_ids=None)

            with activation_hook_context(store=store):
                # max_gen=1 ensures we always run at least the prefill.
                # We don't care about the 1 generated token.
                g.generate(tokens=full_tokens, max_gen=1)

            # Extract decode trajectory from prefill captures.
            # store.get_activations(L) returns [n_full, dim] where n_full = len(full_tokens).
            # h[n_prompt:] are the activations at the positions of generated tokens.
            # Due to causal attention, h[n_prompt + T] during prefill equals what
            # h would be at decode step T+1 — they see the same causal context.
            trajectory: dict[int, torch.Tensor] = {}
            n_actual = len(generated_tokens)

            for layer in layers:
                full = store.get_activations(layer)
                if full is None:
                    continue
                decode_acts = full[n_prompt : n_prompt + n_actual]  # [n_generated, dim]
                if decode_acts.shape[0] == 0:
                    continue
                if stride > 1:
                    decode_acts = decode_acts[::stride]
                trajectory[layer] = decode_acts.half()

            if not trajectory:
                logger.warning(f"Sample {sample.sample_id}: no trajectory captured")
                pbar.update(1)
                continue

            # Update the existing .pt file with the trajectory
            existing["trajectory"] = trajectory
            existing["n_prompt_tokens"] = n_prompt  # overwrite with consistent value

            # Strided token IDs (from re-tokenization)
            decode_token_ids_new = list(generated_tokens[::stride]) if stride > 1 else list(generated_tokens)
            existing["decode_token_ids"] = decode_token_ids_new
            existing["stride"] = stride

            torch.save(existing, act_path)
            n_updated[0] += 1
            pbar.update(1)

        pbar.close()

    except Exception as e:
        exc_queue.put(e)
        logger.exception("Exception in reextract worker")
    finally:
        done_event.set()


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def main(args: BugReExtractArgs) -> None:
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

        # Load all samples, DP-shard them
        pairs = load_pairs(args.pairs_path)
        all_samples = pairs_to_samples(pairs, include_originals=True)
        my_samples = all_samples[dp_rank::n_dp]

        logger.info(
            f"DP rank {dp_rank}/{n_dp}: re-extracting trajectories for "
            f"{len(my_samples)} samples"
        )

        traj_dir = Path(args.traj_dir)

        results: list[dict] = []
        exc_queue: queue.Queue = queue.Queue()
        done_event = threading.Event()
        n_skipped = [0]
        n_updated = [0]

        if is_tp_rank_zero:
            worker = threading.Thread(
                target=run_reextract_worker,
                args=(
                    my_samples, g, traj_dir, args.layers, args.stride,
                    args.track, exc_queue, done_event, n_skipped, n_updated,
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

        if not exc_queue.empty():
            raise RuntimeError("Exception in worker") from exc_queue.get()

        torch.distributed.barrier()

        if is_rank_zero:
            # Recount non-empty trajectories across ALL files
            all_traj = list(traj_dir.glob("trajectories/*.pt"))
            n_with_traj = 0
            n_without_traj = 0
            for pt in all_traj:
                try:
                    d = torch.load(pt, map_location="cpu", weights_only=False)
                    if d.get("trajectory"):
                        n_with_traj += 1
                    else:
                        n_without_traj += 1
                except Exception:
                    pass

            print(f"\nRe-extraction complete:")
            print(f"  Updated: {n_updated[0]} (this DP rank)")
            print(f"  Skipped (already had trajectory): {n_skipped[0]}")
            print(f"  Total trajectories with data: {n_with_traj}/{len(all_traj)}")
            print(f"  Still empty: {n_without_traj}")

    finally:
        uninstall_forward_hooks()

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = load_from_cli(BugReExtractArgs)
    main(args)
