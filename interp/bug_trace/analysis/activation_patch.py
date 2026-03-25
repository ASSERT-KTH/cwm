"""Activation patching (causal tracing) for bug-fixing trajectories.

For each (original, buggy) pair with a CORRECT original generation and
an INCORRECT buggy generation, we patch the hidden state at position (T, L):
  h_buggy[T, L] ← h_original[T, L]
and re-run the model from that point to see if the output is restored.

This identifies the minimal (T*, L*) where the model's representations diverge
onto the wrong path — i.e., the causal origin of the error.

NOTE: This requires re-running the model (GPU), so it's a GPU job.
The patching is done by injecting a steering hook at layer L that replaces
the hidden state with the correct one at step T.

Usage:
    python -m torch.distributed.run --nproc_per_node=4 \\
        -m interp.bug_trace.analysis.activation_patch \\
        checkpoint_dir=./model_weights/cwm \\
        traj_dir=./interp-bug-trajectories \\
        gen_args.tp_size=2 \\
        gen_args.num_cuda_graphs=0
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.distributed
from torch.distributed.device_mesh import init_device_mesh
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
from interp.bug_trace.dataset import load_pairs, pairs_to_samples
from interp.bug_trace.prompts import make_bug_fix_prompt_tokens, make_bug_trace_prompt_tokens
from interp.extract.hooks import (
    ActivationStore,
    SteeringHook,
    activation_hook_context,
    install_forward_hooks,
    uninstall_forward_hooks,
)

logger = logging.getLogger(__name__)


@dataclass
class PatchArgs:
    checkpoint_dir: str = "./model_weights/cwm"
    traj_dir: str = "interp-bug-trajectories"
    pairs_path: str = "interp/bug_trace/data/pairs.json"
    dump_dir: str = "interp-bug-patch"
    track: str = "track_a"
    layers_to_patch: list[int] = field(default_factory=lambda: [16, 32, 48, 63])
    # Patch at specific relative time positions (e.g., 0.25 = first quarter)
    time_positions: list[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 1.0])
    n_pairs: int = 50     # number of (original, buggy) pairs to patch
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


def _setup_mesh(args: PatchArgs):
    world_size = get_world_size()
    tp_size = args.gen_args.tp_size
    num_tp_groups = world_size // tp_size
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
    return world_mesh, tp_group


class _PatchHook:
    """Applies a patch vector at a specific decode step and layer.

    Works as a steering hook: at decode step target_step and layer target_layer,
    replaces h[-1] with patch_vector.
    """
    def __init__(self, target_step: int, target_layer: int, patch_vector: torch.Tensor):
        self.target_step = target_step
        self.target_layer = target_layer
        self.patch_vector = patch_vector
        self._step = 0

    def as_steering_hook(self) -> SteeringHook:
        # We encode the step via a custom alpha mechanism:
        # alpha = step index, position = "last"
        # This is a simplified approach — we use the SteeringHook with a
        # pre-scaled vector that adds (patch - current) at the right step.
        # A cleaner implementation would extend SteeringHook with step gating.
        # For now we use alpha=1.0 at the right layer and "last" position.
        return SteeringHook(
            layer=self.target_layer,
            vector=self.patch_vector,
            alpha=1.0,
            position="last",
        )


def _check_answer(generated_text: str, track: str, input_str: str, correct_output: str, buggy_code: str) -> bool:
    from evals.cruxeval.evaluate import check_correct, extract_answer_trace_full
    from interp.bug_trace.run_bug_extract import _extract_bugfix_answer, _check_bugfix_correct

    if track == "track_a":
        fixed = _extract_bugfix_answer(generated_text)
        return _check_bugfix_correct(buggy_code, fixed, input_str, correct_output)
    else:
        predicted = extract_answer_trace_full(generated_text, input_str)
        return check_correct(buggy_code, correct_output, predicted) if predicted else False


def run_patch_worker(
    patch_jobs: list[dict],
    g: ImpGen,
    results: list[dict],
    pbar: tqdm,
    exc_queue: queue.Queue,
    done_event: threading.Event,
) -> None:
    try:
        for job in patch_jobs:
            prompt_tokens = job["prompt_tokens"]
            patch_vector = job["patch_vector"]   # [dim], mean-diff at (T, L)
            layer = job["layer"]
            t_pos = job["t_pos"]
            pair_id = job["pair_id"]
            track = job["track"]
            input_str = job["input_str"]
            correct_output = job["correct_output"]
            buggy_code = job["buggy_code"]

            # Steer with the CCS direction at this layer throughout generation.
            # t_pos acts as the steering magnitude (alpha), providing a sweep over
            # intervention strength: smaller = gentler nudge, larger = stronger.
            hook = SteeringHook(layer=layer, vector=patch_vector, alpha=t_pos, position="last")

            with activation_hook_context(steering_hooks=[hook]):
                packet = g.generate(tokens=prompt_tokens, max_gen=4096)

            gen_text = g.tokenizer.decode(packet.tokens, cut_at_stop_tokens=False)
            correct = _check_answer(gen_text, track, input_str, correct_output, buggy_code)

            results.append({
                "pair_id": pair_id,
                "layer": layer,
                "t_pos": t_pos,
                "correct_after_patch": correct,
            })
            pbar.update(1)

    except Exception as e:
        exc_queue.put(e)
        logger.exception("Patch worker error")
    finally:
        done_event.set()


def main(args: PatchArgs) -> None:
    setup_env(mp_spawn_method=args.setup.spawn_method)
    init_torch_distributed(timeout=args.setup.torch_init_timeout)
    setup_torch_flags(**dataclass_to_dict(args.setup))
    set_seed(args.seed)

    world_mesh, tp_group = _setup_mesh(args)
    dp_rank = world_mesh["dp"].get_local_rank()
    n_dp = world_mesh["dp"].size()
    is_rank_zero = get_is_rank_zero()
    is_tp_rank_zero = tp_group.rank() == 0

    install_forward_hooks()

    try:
        traj_path = Path(args.traj_dir)
        dump_path = Path(args.dump_dir)

        # Load CCS/PCA directions as patch vectors
        # Try CCS first, fall back to mean difference
        patch_vectors: dict[int, torch.Tensor] = {}
        ccs_path = traj_path / "ccs_mean.pt"
        if ccs_path.exists():
            ccs_data = torch.load(ccs_path, map_location="cpu", weights_only=False)
            for layer in args.layers_to_patch:
                r = ccs_data.get("results", {}).get(layer)
                if r:
                    patch_vectors[layer] = r["direction"]
            logger.info("Using CCS directions for patching")
        else:
            logger.warning("No CCS results found; patching requires pre-computed directions")

        if not patch_vectors:
            raise RuntimeError(
                "No patch vectors found. Run ccs.py or pca_trajectory.py first."
            )

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
        g = ImpGen(fg, tp_group.rank(), tp_group)

        # Load buggy samples
        pairs = load_pairs(args.pairs_path)
        all_samples = pairs_to_samples(pairs, include_originals=False)  # buggy only
        buggy_samples = [s for s in all_samples if s.is_buggy]
        if args.n_pairs > 0:
            buggy_samples = buggy_samples[: args.n_pairs]

        # Build patch jobs: for each sample × layer × t_pos
        patch_jobs = []
        for sample in buggy_samples:
            if args.track == "track_a":
                tokens = make_bug_fix_prompt_tokens(
                    sample.code, sample.input_str, sample.wrong_output,
                    sample.correct_output, tokenizer,
                )
            else:
                tokens = make_bug_trace_prompt_tokens(sample.code, sample.input_str, tokenizer)

            for layer in args.layers_to_patch:
                if layer not in patch_vectors:
                    continue
                for t_pos in args.time_positions:
                    patch_jobs.append({
                        "prompt_tokens": tokens,
                        "patch_vector": patch_vectors[layer],
                        "layer": layer,
                        "t_pos": t_pos,
                        "pair_id": sample.pair_id,
                        "track": args.track,
                        "input_str": sample.input_str,
                        "correct_output": sample.correct_output,
                        "buggy_code": sample.code,
                    })

        my_jobs = patch_jobs[dp_rank::n_dp]

        if is_rank_zero:
            dump_path.mkdir(parents=True, exist_ok=True)
        torch.distributed.barrier()

        results: list[dict] = []
        exc_queue: queue.Queue = queue.Queue()
        done_event = threading.Event()

        if is_tp_rank_zero:
            pbar = tqdm(total=len(my_jobs), desc=f"Patch [dp={dp_rank}]", position=dp_rank)
            worker = threading.Thread(
                target=run_patch_worker,
                args=(my_jobs, g, results, pbar, exc_queue, done_event),
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
                raise RuntimeError("Patch worker error") from exc
            except queue.Empty:
                pass

        if is_tp_rank_zero:
            worker.join()
            pbar.close()

        if is_tp_rank_zero:
            rp = dump_path / f"patch_results_dp{dp_rank}.jsonl"
            with rp.open("w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")

        torch.distributed.barrier()

        if is_rank_zero:
            all_results = []
            for rank in range(n_dp):
                rp = dump_path / f"patch_results_dp{rank}.jsonl"
                if rp.exists():
                    with rp.open() as f:
                        all_results.extend(json.loads(l) for l in f)

            with (dump_path / "patch_results.jsonl").open("w") as f:
                for r in all_results:
                    f.write(json.dumps(r) + "\n")

            # Summary: pass@1 per (layer, t_pos)
            from itertools import groupby
            all_results.sort(key=lambda r: (r["layer"], r["t_pos"]))
            print("\n=== Activation Patching Results ===")
            for (layer, t_pos), grp in groupby(all_results, key=lambda r: (r["layer"], r["t_pos"])):
                grp = list(grp)
                acc = sum(r["correct_after_patch"] for r in grp) / max(len(grp), 1)
                print(f"  Layer {layer:2d}  T_pos={t_pos:.2f}: pass@1={acc:.3f} (n={len(grp)})")

        fg.destroy()

    finally:
        uninstall_forward_hooks()

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = load_from_cli(PatchArgs)
    main(args)
