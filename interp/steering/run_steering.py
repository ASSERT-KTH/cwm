"""Run CruxEval with activation steering interventions.

Example (8 GPUs, TP=2, DP=4):

    python -m torch.distributed.run --nproc_per_node=8 \\
        -m interp.steering.run_steering \\
        checkpoint_dir=./model_weights/cwm \\
        vector_dir=./interp-vectors \\
        condition=correct_vs_incorrect \\
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
    SteeringHook,
    activation_hook_context,
    install_forward_hooks,
    uninstall_forward_hooks,
)
from interp.steering.vectors import (
    compute_steering_vectors,
    load_steering_vectors,
    save_steering_vectors,
)

logger = logging.getLogger(__name__)

_MAX_GEN: dict[str, int] = {
    "direct": 512,
    "trace_full": 8192,
    "trace_single_step": 512,
}


@dataclass
class SteeringArgs:
    checkpoint_dir: str = ""
    dump_dir: str = "interp-steer"
    extract_dir: str = ""     # Phase-1 output, used to compute vectors if vector_dir empty
    vector_dir: str = ""      # Pre-computed vectors (.pt)
    condition: str = "correct_vs_incorrect"
    target_layers: list[int] = field(default_factory=lambda: [32, 40, 48, 56, 63])
    alphas: list[float] = field(
        default_factory=lambda: [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    )
    mode: str = "trace_full"
    n_samples: int = 200
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


def setup_mesh(args: SteeringArgs) -> tuple[DeviceMesh, torch.distributed.ProcessGroup]:
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


def run_steer_worker(
    samples: list[dict],
    g: ImpGen,
    layer: int,
    alpha: float,
    vector: torch.Tensor,
    mode: str,
    max_gen: int,
    results: list[dict],
    pbar: tqdm,
    exc_queue: queue.Queue,
    done_event: threading.Event,
) -> None:
    hook = SteeringHook(layer=layer, vector=vector, alpha=alpha, position="all")
    try:
        for sample in samples:
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

            gen_kwargs: dict = dict(max_gen=max_gen)
            if stop_str:
                gen_kwargs["stop_str"] = stop_str

            steering_hooks = [hook] if alpha != 0.0 else []
            with activation_hook_context(steering_hooks=steering_hooks):
                packet = g.generate(tokens=prompt_tokens, **gen_kwargs)

            generation = g.tokenizer.decode(packet.tokens, cut_at_stop_tokens=False)
            predicted = extract_fn(generation)
            correct = (
                check_correct(code, sample["output"], predicted)
                if predicted is not None
                else False
            )
            results.append(
                {
                    "id": sample["id"],
                    "layer": layer,
                    "alpha": alpha,
                    "expected": sample["output"],
                    "predicted": predicted,
                    "correct": correct,
                }
            )
            pbar.update(1)
    except Exception as e:
        exc_queue.put(e)
        logger.exception("Exception in steer worker")
    finally:
        done_event.set()


def _run_one_sweep(
    samples: list[dict],
    g: ImpGen,
    layer: int,
    alpha: float,
    vector: torch.Tensor,
    mode: str,
    max_gen: int,
    dp_rank: int,
) -> list[dict]:
    results: list[dict] = []
    exc_queue: queue.Queue = queue.Queue()
    done_event = threading.Event()
    is_tp_rank_zero = g.tp_rank == 0

    if is_tp_rank_zero:
        pbar = tqdm(
            total=len(samples),
            desc=f"Steer layer={layer} α={alpha:.2f} [dp={dp_rank}]",
            position=dp_rank,
            leave=False,
        )
        worker = threading.Thread(
            target=run_steer_worker,
            args=(
                samples,
                g,
                layer,
                alpha,
                vector,
                mode,
                max_gen,
                results,
                pbar,
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
            raise RuntimeError("Exception in steer worker") from exc
        except queue.Empty:
            pass

    if is_tp_rank_zero:
        worker.join()
        pbar.close()

    if not exc_queue.empty():
        raise RuntimeError("Exception in steer worker") from exc_queue.get()

    return results


def main(args: SteeringArgs) -> None:
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
        # Load or compute steering vectors on rank 0, then broadcast
        if is_rank_zero:
            if args.vector_dir:
                vectors = load_steering_vectors(
                    str(Path(args.vector_dir) / "vectors.pt")
                )
            else:
                assert args.extract_dir, "Need extract_dir or vector_dir"
                vectors = compute_steering_vectors(
                    extract_dir=args.extract_dir,
                    condition=args.condition,
                    layers=args.target_layers,
                )
                vdir = Path(args.dump_dir) / "vectors"
                vdir.mkdir(parents=True, exist_ok=True)
                save_steering_vectors(
                    vectors,
                    str(vdir / "vectors.pt"),
                    condition=args.condition,
                )
        else:
            vectors = {}

        # Broadcast vectors to all ranks via CPU gather (small tensors)
        for layer in args.target_layers:
            if is_rank_zero:
                v = vectors.get(layer, torch.zeros(1))
                shape = torch.tensor(list(v.shape), dtype=torch.long)
            else:
                shape = torch.zeros(1, dtype=torch.long)

            shape_list = [torch.zeros_like(shape) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(shape_list, shape)
            true_shape = shape_list[0].tolist()

            if is_rank_zero:
                buf = vectors.get(layer, torch.zeros(*true_shape))
            else:
                buf = torch.zeros(*true_shape)
            torch.distributed.broadcast(buf, src=0)
            if not is_rank_zero:
                vectors[layer] = buf

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

        dump_path = Path(args.dump_dir)
        if is_rank_zero:
            dump_path.mkdir(parents=True, exist_ok=True)
        torch.distributed.barrier()

        max_gen = _MAX_GEN.get(args.mode, 4096)
        all_sweep_results: list[dict] = []

        for layer in args.target_layers:
            vec = vectors.get(layer)
            if vec is None:
                logger.warning(f"No vector for layer {layer}, skipping")
                continue

            for alpha in args.alphas:
                results = _run_one_sweep(
                    my_samples, g, layer, alpha, vec, args.mode, max_gen, dp_rank
                )
                if g.tp_rank == 0:
                    all_sweep_results.extend(results)

        # Write per-rank results
        if g.tp_rank == 0:
            rank_path = dump_path / f"sweep_dp{dp_rank}.jsonl"
            with rank_path.open("w") as f:
                for r in all_sweep_results:
                    f.write(json.dumps(r) + "\n")

        torch.distributed.barrier()

        if is_rank_zero:
            combined: list[dict] = []
            for rank in range(n_dp):
                rp = dump_path / f"sweep_dp{rank}.jsonl"
                if rp.exists():
                    with rp.open() as f:
                        combined.extend(json.loads(l) for l in f)

            with (dump_path / "sweep_results.jsonl").open("w") as f:
                for r in combined:
                    f.write(json.dumps(r) + "\n")

            # Summary: pass@1 per (layer, alpha)
            from itertools import groupby

            combined.sort(key=lambda r: (r["layer"], r["alpha"]))
            summary_rows = []
            for (layer, alpha), grp in groupby(
                combined, key=lambda r: (r["layer"], r["alpha"])
            ):
                grp = list(grp)
                p1 = sum(r["correct"] for r in grp) / len(grp) if grp else 0.0
                summary_rows.append(
                    {"layer": layer, "alpha": alpha, "pass_at_1": p1, "n": len(grp)}
                )
                print(f"layer={layer:2d} alpha={alpha:+.2f}  pass@1={p1:.4f}  (n={len(grp)})")

            with (dump_path / "sweep_summary.jsonl").open("w") as f:
                for r in summary_rows:
                    f.write(json.dumps(r) + "\n")

            try:
                import wandb

                if args.wandb_project:
                    wandb.init(
                        project=args.wandb_project,
                        name=args.wandb_run_name or f"steer-{args.condition}",
                        config=dataclass_to_dict(args),
                    )
                    table = wandb.Table(
                        columns=["layer", "alpha", "pass_at_1"],
                        data=[[r["layer"], r["alpha"], r["pass_at_1"]] for r in summary_rows],
                    )
                    wandb.log({"steering_sweep": table})
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
    args = load_from_cli(SteeringArgs)
    main(args)
