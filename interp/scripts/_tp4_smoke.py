"""Smoke test: load CWM with TP=4 and generate a few tokens.

Reports peak GPU memory usage to confirm it fits on 40GB nodes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
import torch.distributed
from torch.distributed.device_mesh import init_device_mesh

from cwm.common.environment import (
    get_is_rank_zero,
    get_world_size,
    init_torch_distributed,
    set_seed,
    setup_env,
    setup_torch_flags,
)
from cwm.common.params import load_from_cli
from cwm.fastgen.generate import FastGen
from cwm.fastgen.utils.loading import build_fastgen_model, build_tokenizer_from_ckpt
from cwm.rl.lib.impgen import ImpGen
from evals.args import FastGenArgs, SetupArgs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SmokeArgs:
    checkpoint_dir: str = "./model_weights/cwm"
    tp_size: int = 4
    setup: SetupArgs = field(default_factory=SetupArgs)


def main(args: SmokeArgs) -> None:
    setup_env()
    init_torch_distributed(timeout=args.setup.torch_init_timeout)
    setup_torch_flags()
    set_seed(42)

    world_size = get_world_size()
    assert world_size == args.tp_size, f"Expected {args.tp_size} ranks, got {world_size}"

    world_mesh = init_device_mesh(
        device_type="cuda",
        mesh_shape=(1, args.tp_size),
        mesh_dim_names=("dp", "tp"),
    )
    global_rank = world_mesh.get_rank()
    tp_group = None
    for ranks in world_mesh.mesh.tolist():
        pg = torch.distributed.new_group(ranks, backend="moodist")
        if global_rank in ranks:
            tp_group = pg

    is_rank_zero = get_is_rank_zero()

    if is_rank_zero:
        logger.info(f"Loading model with TP={args.tp_size}...")

    torch.cuda.reset_peak_memory_stats()

    tokenizer = build_tokenizer_from_ckpt(args.checkpoint_dir)
    model = build_fastgen_model(
        world_mesh=world_mesh,
        checkpoint_dir=args.checkpoint_dir,
    )

    gen_args = FastGenArgs(
        tp_size=args.tp_size,
        use_sampling=False,
        temperature=0.0,
        num_cuda_graphs=0,
    )
    fg = FastGen(
        gen_args,
        model=model,
        tokenizer=tokenizer,
        dtype=torch.bfloat16,
        device=torch.device(f"cuda:{torch.cuda.current_device()}"),
        tp_mesh=world_mesh["tp"],
    )
    g = ImpGen(fg, tp_group.rank(), tp_group)

    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    total_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2

    if is_rank_zero:
        logger.info(f"Model loaded. Peak memory after load: {peak_mb:.0f} MB / {total_mb:.0f} MB")

    # Generate a few tokens to confirm inference works
    import threading, queue as q_mod
    prompt = tokenizer.encode("def f(x):\n    return x + 1\n")
    result_q: queue.Queue = q_mod.Queue()

    def _worker():
        try:
            packet = g.generate(tokens=prompt, max_gen=20)
            result_q.put(packet)
        except Exception as e:
            result_q.put(e)

    if tp_group.rank() == 0:
        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    while True:
        done = g.work()
        if done:
            break
        if not result_q.empty():
            g.stop()

    peak_mb_after = torch.cuda.max_memory_allocated() / 1024**2

    if is_rank_zero:
        result = result_q.get_nowait() if not result_q.empty() else None
        if isinstance(result, Exception):
            logger.error(f"Generation failed: {result}")
        else:
            gen_text = tokenizer.decode(result.tokens) if result else "(no result)"
            logger.info(f"Generated: {gen_text!r}")
        logger.info(f"Peak memory after generation: {peak_mb_after:.0f} MB / {total_mb:.0f} MB ({peak_mb_after/total_mb*100:.1f}%)")
        print(f"\nRESULT: TP=4 peak_memory={peak_mb_after:.0f}/{total_mb:.0f} MB ({peak_mb_after/total_mb*100:.1f}%)")
        if peak_mb_after < total_mb * 0.9:
            print("STATUS: OK — fits on 40GB node")
        else:
            print("STATUS: TIGHT — may OOM under load")

    fg.destroy()
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    args = load_from_cli(SmokeArgs)
    main(args)
