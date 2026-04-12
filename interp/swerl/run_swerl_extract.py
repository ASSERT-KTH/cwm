# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Activation extraction runner for SWEbench trajectories.

Mirrors evals/main.py but wraps each g.generate() call with
activation_hook_context so that hidden states are captured at every decode
step.  Saves per-trajectory .pt files for downstream probe training.

Usage (2-GPU run for testing, 8-GPU for full extraction):

    python -m torch.distributed.run --nproc_per_node=2 \\
        -m interp.swerl.run_swerl_extract -- \\
        config=eval_sbv_extract.yaml \\
        dump_dir=interp-swerl-extract \\
        checkpoint_dir=./model_weights/cwm \\
        layers=[32] \\
        n_instances=150

The config file is the same format as eval_sbv_1instance.yaml.
Additional CLI args (overriding defaults in ExtractArgs):
    layers         — list of layer indices to capture (default: [32])
    n_instances    — number of instances to extract (default: 150)
    stride         — capture every Nth decode token (default: 1, i.e. all)
"""

from __future__ import annotations

import json
import logging
import queue
import re
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import moodist
import torch
import torch.distributed
from torch.distributed.device_mesh import DeviceMesh

import cwm.rl.envs.envs
import cwm.text.data
from cwm.common.environment import (
    get_is_rank_zero,
    get_world_size,
    init_torch_distributed,
    set_seed,
    setup_env,
    setup_torch_flags,
)
from cwm.common.params import dataclass_to_dict, load_from_cli, save_params
from cwm.data.dataset import Dataset
from cwm.fastgen.generate import FastGen
from cwm.fastgen.utils.loading import build_fastgen_model, build_tokenizer_from_ckpt
from cwm.logging.logger import add_logger_file_handler, initialize_logger, set_root_log_level
from cwm.rl.envs import Trajectory, build_env_overrides, get_reward_fn
from cwm.rl.envs.api import Env, RewardFn
from cwm.rl.envs.config import TaskIdxDatum, to_task_idx_datum
from cwm.rl.lib.datatypes import DataSource
from cwm.rl.lib.impgen import ImpGen
from cwm.text.datatypes import BaseTextDatum
from evals.args import RLEvalArgs, FastGenArgs, SetupArgs
from functools import partial

from interp.extract.hooks import (
    ActivationStore,
    activation_hook_context,
    install_forward_hooks,
)
from interp.swerl.labels import build_label_arrays, classify_tool_call

# Serialises activation_hook_context usage within a process.
# Multiple rollout threads share one ImpGen and one process-global _current_store.
# Without this lock, concurrent threads clobber each other's store, producing
# 0-capture trajectories.  The lock only holds during g.generate() (GPU-bound);
# env.step() (Modal I/O-bound) runs fully concurrently outside it.
_hook_lock = threading.Lock()

Dataset.register_package(cwm.text.data, ["srcs"])

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_EXTRACT_LAYERS_DEFAULT = [32]
_CAPTURE_STRIDE_DEFAULT = 1   # capture every token within an action


@dataclass
class ExtractArgs(RLEvalArgs):
    """Extends RLEvalArgs with extraction-specific options."""

    # Layers to capture (subset of [0..63])
    layers: list[int] = field(default_factory=lambda: list(_EXTRACT_LAYERS_DEFAULT))
    # Activation capture stride within each action (1 = every token)
    stride: int = _CAPTURE_STRIDE_DEFAULT
    # Maximum number of trajectories to extract (-1 = all)
    n_instances: int = -1
    # Skip instances whose .pt file already exists in save_dir (for resuming)
    resume: bool = True


# ---------------------------------------------------------------------------
# Tool-call parser
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(
    r"<tool:\s*(?P<name>\w+)\s*>(?P<input>.*?)</tool>",
    re.DOTALL,
)


def _parse_last_tool_call(text: str) -> tuple[str, str]:
    """Return (tool_name, tool_input) of the last tool call in text.

    Falls back to ("unknown", "") if no tool call is found.
    """
    matches = list(_TOOL_CALL_RE.finditer(text))
    if not matches:
        return "unknown", ""
    m = matches[-1]
    return m.group("name").strip(), m.group("input").strip()


# ---------------------------------------------------------------------------
# Extraction rollout
# ---------------------------------------------------------------------------


def _rollout_with_capture(
    args: ExtractArgs,
    env: Env,
    rewardfn: RewardFn,
    g: ImpGen,
    start_args: dict,
    save_dir: Path,
    instance_id: str,
) -> dict | None:
    """Run one trajectory with activation capture.

    Returns a metadata dict (without the large activation tensors) on success,
    or None on failure.  The full .pt file is written to save_dir.
    """
    store = ActivationStore(
        layers=args.layers,
        capture_token_ids=None,  # capture all positions
    )

    traj = Trajectory()
    turn_log: list[dict] = []
    total_tokens = 0

    try:
        state, tr = env.start(start_args)
        with state:
            traj.append(tr)
            turn_idx = 0
            while not tr.terminal and len(traj.context) < 131072:
                # Set the current turn, then exclusively own _current_store
                # for the duration of this generate call.  The lock prevents
                # concurrent rollout threads from clobbering each other's store.
                # env.step() (Modal I/O) runs outside the lock — no slowdown there.
                store.current_turn = turn_idx
                max_gen = env.max_action_len(state)
                store.seqlen_start = len(traj.context)
                store.seqlen_end = len(traj.context) + max_gen
                with _hook_lock, activation_hook_context(store=store):
                    action_packet = g.generate(
                        tokens=traj.context,
                        max_gen=max_gen,
                        temperature=None,
                        stop_str=getattr(env, "stop_str", None),
                    )
                action_tokens = action_packet.tokens

                # Parse tool call from decoded action text
                action_text = g.tokenizer.decode(action_tokens)
                tool_name, tool_input = _parse_last_tool_call(action_text)
                stage = classify_tool_call(tool_name, tool_input)
                turn_log.append({
                    "turn_idx": turn_idx,
                    "tool_name": tool_name,
                    "tool_input": tool_input[:300],
                    "stage": stage,
                })

                tr = env.step(state, action_tokens)
                tr.add_rewards(rewardfn(tr))
                traj.append(tr)
                total_tokens = len(traj.context)
                turn_idx += 1

    except Exception:
        logger.exception(f"Exception during extraction rollout for {instance_id}")
        return None

    outcome: bool = bool(traj.transitions[-1].outcomes.get("pass", False))

    # Build label arrays
    labels = build_label_arrays(
        turn_log=turn_log,
        turn_indices=store.turn_indices,
        positions=store.positions,
        outcome=outcome,
    )

    # Apply stride subsampling (keep every Nth captured token)
    stride = args.stride
    positions_strided = store.positions[::stride]
    turn_indices_strided = store.turn_indices[::stride]
    activations_strided = {
        layer: store.get_activations(layer)[::stride].half()  # save as fp16
        for layer in args.layers
        if store.get_activations(layer) is not None
    }
    labels_strided = {
        k: v[::stride] if isinstance(v, list) else v
        for k, v in labels.items()
    }

    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{instance_id}.pt"
    torch.save(
        {
            "instance_id": instance_id,
            "outcome": outcome,
            "n_turns": turn_idx,
            "turn_log": turn_log,
            "total_tokens": total_tokens,
            "positions": positions_strided,
            "turn_indices": turn_indices_strided,
            "labels": labels_strided,
            "activations": activations_strided,
            "stride": stride,
            "layers": args.layers,
        },
        out_path,
    )
    logger.info(
        f"Saved extraction for {instance_id}: outcome={outcome}, "
        f"turns={turn_idx}, captures={len(positions_strided)}, path={out_path}"
    )
    return {"instance_id": instance_id, "outcome": outcome, "n_turns": turn_idx}


# ---------------------------------------------------------------------------
# Rollout thread (mirrors evals/main.py::rollout)
# ---------------------------------------------------------------------------


def rollout_extract(
    args: ExtractArgs,
    environments_and_rewards: list[tuple[Env, Callable[..., RewardFn]]],
    g: ImpGen,
    data_queue: moodist.Queue,
    done_queue: queue.Queue,
    dump_queue: moodist.Queue,
    save_dir: Path,
) -> None:
    while True:
        data = data_queue.get_object()
        if data is None:
            done_queue.put(True)
            dump_queue.put_object(None)
            logger.info("Extract rollout thread received kill signal")
            return

        assert isinstance(data, TaskIdxDatum)
        task_args, env_idx, start_args = data.val
        env, rewardfn_ctor = environments_and_rewards[env_idx]
        rewardfn = rewardfn_ctor()
        instance_id = start_args.get("instance_id", str(data.src))

        logger.info(f"Extracting {instance_id}")
        meta = _rollout_with_capture(args, env, rewardfn, g, start_args, save_dir, instance_id)
        if meta is not None:
            dump_queue.put_object(meta)
        else:
            logger.warning(f"Extraction failed for {instance_id}, skipping")
        logger.info(f"Done extracting {instance_id}")


# ---------------------------------------------------------------------------
# Main extraction loop (mirrors evals/main.py structure)
# ---------------------------------------------------------------------------


def run_extraction(args: ExtractArgs) -> None:
    assert args.dump_dir, "dump_dir must be set"
    dump_path = Path(args.dump_dir)
    save_dir = dump_path / "activations"

    setup_env(mp_spawn_method=args.setup.spawn_method)
    init_torch_distributed(timeout=args.setup.torch_init_timeout)
    setup_torch_flags(**dataclass_to_dict(args.setup))
    set_seed(args.seed)

    from evals.main import setup_mesh, thread_wrapper

    world_mesh, all_group, tp_group = setup_mesh(args)
    global_rank = world_mesh.get_rank()

    if get_is_rank_zero():
        dump_path.mkdir(parents=True, exist_ok=True)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_params(args, dump_path / "extract_config.yaml")
    add_logger_file_handler(dump_path / "extract.log")

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

    install_forward_hooks()

    runtime_kwargs = {"tokenizer": g.tokenizer}
    environments_and_rewards = [
        (build_env_overrides(task, **runtime_kwargs), get_reward_fn(task.reward_fn))
        for task in args.tasks
    ]

    data_queue = moodist.Queue(all_group, location=0)
    dump_queue = moodist.Queue(all_group, location=0)
    done_queue: queue.Queue = queue.Queue()
    exc_queue: queue.Queue = queue.Queue()

    loading_thread = None
    if global_rank == 0:
        task_datasets = {}
        for env_idx, task in enumerate(args.tasks):
            assert task.path is not None
            dataset_path = Path(task.path)
            to_datum = partial(to_task_idx_datum, task, env_idx)

            # Map first (produces TaskIdxDatum with .val/.src), then filter
            all_items = list(Dataset.from_jsonl(dataset_path).map(to_datum))

            # Resume: skip instances whose .pt file already exists
            if args.resume:
                n_before = len(all_items)
                all_items = [
                    item for item in all_items
                    if not (save_dir / f"{item.val[2].get('instance_id', str(item.src))}.pt").exists()
                ]
                n_skipped = n_before - len(all_items)
                logger.info(
                    f"Resume: skipped {n_skipped} already-extracted instances, "
                    f"{len(all_items)} remaining"
                )

            # Limit to n_instances if set (applied after resume filter)
            if args.n_instances > 0:
                all_items = all_items[: args.n_instances]

            ds = Dataset.from_list(all_items)
            task_datasets[task.name] = ds

        dataset = Dataset.chain(list(task_datasets.values()))

        from evals.main import load_data
        loading_thread = threading.Thread(
            target=thread_wrapper,
            kwargs=dict(
                target=load_data,
                dump_samples_path=dump_path,
                dataset=dataset,
                data_queue=data_queue,
                num_rollout_threads=args.num_rollout_threads,
                exc_queue=exc_queue,
            ),
        )
        loading_thread.start()

    rollout_threads = []
    for _ in range(args.num_rollout_threads):
        t = threading.Thread(
            target=thread_wrapper,
            kwargs=dict(
                target=rollout_extract,
                args=args,
                environments_and_rewards=environments_and_rewards,
                g=g,
                data_queue=data_queue,
                done_queue=done_queue,
                dump_queue=dump_queue,
                save_dir=save_dir,
                exc_queue=exc_queue,
            ),
        )
        rollout_threads.append(t)
        t.start()

    # Dump thread: write index.jsonl
    index_file = dump_path / "index.jsonl"
    dump_thread = None

    def _dump(dump_queue: moodist.Queue, n_threads: int, world_size: int) -> None:
        kill_counts = 0
        with index_file.open("a") as f:
            while True:
                item = dump_queue.get_object()
                if item is None:
                    kill_counts += 1
                    if kill_counts >= n_threads * world_size:
                        return
                    continue
                f.write(json.dumps(item) + "\n")
                f.flush()

    if global_rank == 0:
        dump_thread = threading.Thread(
            target=_dump,
            args=(dump_queue, args.num_rollout_threads, get_world_size()),
        )
        dump_thread.start()

    # Main generation loop
    done = False
    while not done:
        done = g.work()
        if done_queue.qsize() >= args.num_rollout_threads:
            g.stop()
        try:
            ex = exc_queue.get_nowait()
        except queue.Empty:
            pass
        else:
            raise RuntimeError("Exception in extraction thread") from ex

    if loading_thread:
        loading_thread.join()
    for t in rollout_threads:
        t.join()
    fg.destroy()
    if dump_thread:
        dump_thread.join()

    # Shutdown
    shutdown_queue = moodist.Queue(all_group, location=0)
    if get_is_rank_zero():
        for _ in range(get_world_size()):
            shutdown_queue.put_object(None)
    shutdown_queue.get_object()

    torch.distributed.barrier()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    if get_is_rank_zero():
        n = sum(1 for _ in open(index_file)) if index_file.exists() else 0
        logger.info(f"Extraction complete: {n} trajectories saved to {save_dir}")


def main() -> None:
    initialize_logger()
    args = load_from_cli(ExtractArgs, from_config_file=True, with_preset=True)
    set_root_log_level(args.log_level)
    run_extraction(args)


if __name__ == "__main__":
    main()
