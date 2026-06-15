# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
CRUXEval-O output prediction for a CODI-distilled CWM model.

Predicts the *execution result* (CRUXEval-O pass@1), not the trace-quality
metrics of ``evals.trace_analysis``. Supports the same four prompt modes as
``evals.cruxeval.run_eval``: direct, reasoning, trace_full, trace_single_step.
The CODI student was distilled on execution traces (``cwm.training.data`` uses
the full-trace prompt), so ``trace_full`` is the native CODI eval mode.

Loads a HuggingFace CWM base + a trained CODI LoRA adapter + ``thought_projector.pt``
(``cwm.training.train`` output). Whenever the model emits ``<|line_sep|>`` it
injects ``latent_steps`` continuous thoughts in place of the per-frame locals,
mirroring the no-grad path of ``cwm.training.codi_streaming``. Sampling
defaults match ``evals.cruxeval.run_eval`` (10 generations, temperature=0.6,
top_p=0.95); set temperature<=0 for greedy decoding.

    torchrun --nproc_per_node=8 -m evals.cruxeval.run_eval_codi \\
        --adapter_dir /path/to/codi_qlora_..._output_dir \\
        --base_model model_weights/cwm_hf \\
        --latent_steps 2 \\
        --dump_dir eval-codi-cruxeval
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from transformers.cache_utils import DynamicCache

from cwm.training.codi_config import default_codi_config_from_tokenizer
from dataset.sources import load_rows
from evals.cruxeval.evaluate import (
    check_correct,
    extract_answer,
    extract_answer_reasoning,
    extract_answer_trace_full,
    extract_answer_trace_single_step,
)
from evals.cruxeval.prompts import (
    REASONING_SYSTEM_PROMPT,
    _make_trace_context,
    make_direct_output_prompt,
)

logger = logging.getLogger(__name__)

_MODES = ("direct", "reasoning", "trace_full", "trace_single_step")

# Visible-token budgets aligned with evals.cruxeval.run_eval.
_MAX_GEN: dict[str, int] = {
    "direct": 512,
    "reasoning": 8192,
    "trace_full": 8192,
    "trace_single_step": 512,
}


def _token_id(tok, token: str) -> int:
    token_id = tok.convert_tokens_to_ids(token)
    if token_id is None or token_id == tok.unk_token_id:
        encoded = tok.encode(token, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"{token!r} is not a single token: {encoded}")
        token_id = encoded[0]
    return int(token_id)


def build_trace_full_prompt_ids(code: str, input_str: str, tok) -> list[int]:
    """HF-tokenizer port of ``make_trace_full_prompt_tokens`` (the seeded
    full-trace prompt the CODI student was trained on):
    ``[BOS][TRACE_CONTEXT_START]$CONTEXT[FRAME_SEP][CALL_SEP]{}
    [ACTION_SEP]def main():\n[FRAME_SEP]``.
    """
    context = _make_trace_context(code, input_str)
    ids = [tok.bos_token_id, _token_id(tok, "<|trace_context_start|>")]
    ids += tok.encode(context, add_special_tokens=False)
    ids += [_token_id(tok, "<|frame_sep|>"), _token_id(tok, "<|call_sep|>")]
    ids += tok.encode("{}", add_special_tokens=False)
    ids += [_token_id(tok, "<|action_sep|>")]
    ids += tok.encode("def main():\n", add_special_tokens=False)
    ids += [_token_id(tok, "<|frame_sep|>")]
    return ids


def build_trace_single_step_prompt_ids(code: str, input_str: str, tok) -> list[int]:
    """HF-tokenizer port of ``make_trace_single_step_prompt_tokens``."""
    ids = build_trace_full_prompt_ids(code, input_str, tok)
    ids.append(_token_id(tok, "<|return_sep|>"))
    return ids


def build_reasoning_prompt_ids(code: str, input_str: str, tok) -> list[int]:
    """HF-tokenizer port of ``make_reasoning_prompt_tokens``."""
    base = make_direct_output_prompt(code, input_str)
    assert base.endswith("[ANSWER]\n"), "unexpected prompt suffix"
    user_msg = base[: -len("[ANSWER]\n")]

    ids = [tok.bos_token_id]
    ids += [_token_id(tok, "<|start_header_id|>")]
    ids += tok.encode("system", add_special_tokens=False)
    ids += [_token_id(tok, "<|end_header_id|>")]
    ids += tok.encode("\n\n", add_special_tokens=False)
    ids += tok.encode(REASONING_SYSTEM_PROMPT, add_special_tokens=False)
    ids += [_token_id(tok, "<|eot_id|>")]
    ids += [_token_id(tok, "<|start_header_id|>")]
    ids += tok.encode("user", add_special_tokens=False)
    ids += [_token_id(tok, "<|end_header_id|>")]
    ids += tok.encode("\n\n", add_special_tokens=False)
    ids += tok.encode(user_msg, add_special_tokens=False)
    ids += [_token_id(tok, "<|eot_id|>")]
    ids += [_token_id(tok, "<|start_header_id|>")]
    ids += tok.encode("assistant", add_special_tokens=False)
    ids += [_token_id(tok, "<|end_header_id|>")]
    ids += tok.encode("\n\n", add_special_tokens=False)
    ids += tok.encode("<think>\n", add_special_tokens=False)
    return ids


def build_prompt_ids(mode: str, code: str, input_str: str, tok) -> list[int]:
    if mode == "direct":
        ids = [] if tok.bos_token_id is None else [int(tok.bos_token_id)]
        ids += tok.encode(
            make_direct_output_prompt(code, input_str), add_special_tokens=False
        )
        return ids
    if mode == "reasoning":
        return build_reasoning_prompt_ids(code, input_str, tok)
    if mode == "trace_full":
        return build_trace_full_prompt_ids(code, input_str, tok)
    if mode == "trace_single_step":
        return build_trace_single_step_prompt_ids(code, input_str, tok)
    raise ValueError(f"Unknown mode: {mode!r}")


_EXTRACTORS = {
    "direct": extract_answer,
    "reasoning": extract_answer_reasoning,
    "trace_full": extract_answer_trace_full,
    "trace_single_step": extract_answer_trace_single_step,
}


def make_batches(
    kept: list[tuple[int, dict, list[int]]],
    batch_size: int,
) -> list[list[tuple[int, dict, list[int]]]]:
    n_batches = (len(kept) + batch_size - 1) // batch_size
    batches = [[] for _ in range(n_batches)]
    # Spread the longest prompts across different batches so the shared queue
    # does not end with one pathological all-long batch.
    for i, item in enumerate(sorted(kept, key=lambda x: x[0], reverse=True)):
        batches[i % n_batches].append(item)
    return [b for b in batches if b]


def build_thought_projector(hidden_size: int, device, dtype) -> nn.Sequential:
    """Same architecture as ``CodiModel.__init__`` so the state_dict lines up."""
    proj = nn.Sequential(
        nn.Linear(hidden_size, hidden_size, bias=False),
        nn.SiLU(),
        nn.Linear(hidden_size, hidden_size, bias=False),
        nn.LayerNorm(hidden_size),
    )
    return proj.to(device=device, dtype=dtype)


class CodiGenerator:
    """Batched sampling generation with CODI latent injection.

    Lock-step over the batch: every lane feeds exactly one embedding per forward
    and advances one KV position, so all lanes stay cache-aligned. On
    ``<|line_sep|>`` a lane's per-frame locals are replaced by a fixed feed
    schedule (latent-start, ``latent_steps`` projected thoughts, latent-end)
    before the next visible token is sampled -- mirroring the no-grad path of
    ``cwm.training.codi_streaming``. Prompts are left-padded so lanes of unequal
    length share one rectangular forward.

    A "feed" is a token id, or ``None`` meaning "project the lane's previous
    hidden state" (a latent thought).
    """

    def __init__(self, student, projector, cfg, device) -> None:
        assert (
            cfg.latent_start_token_id is not None
            and cfg.latent_end_token_id is not None
        )
        self.lm = student.base_model.model  # CwmForCausalLM
        self.embed = student.get_input_embeddings()
        self.proj = projector
        self.cfg = cfg
        self.device = device
        self.dtype = self.embed.weight.dtype

    def _schedule(self, token: int) -> list[int | None]:
        """Feeds to run after sampling ``token`` to reach the next visible logits.

        This mirrors ``_student_spans`` in training: ``<|line_sep|>`` remains a
        visible token, the locals text up to ``<|action_sep|>`` is replaced by
        latent-start/thoughts/latent-end, and only then do we sample the next
        visible token.
        """
        c = self.cfg
        if token == c.latent_span_start_token_id:
            return [
                token,
                c.latent_start_token_id,
                *([None] * c.latent_steps),
                c.latent_end_token_id,
            ]
        return [token]

    @staticmethod
    def _endswith(tokens: list[int], suffix: list[int]) -> bool:
        return (
            bool(suffix)
            and len(tokens) >= len(suffix)
            and tokens[-len(suffix) :] == suffix
        )

    @staticmethod
    def _sample_next(
        logits: torch.Tensor, temperature: float, top_p: float
    ) -> torch.Tensor:
        if temperature <= 0:
            return logits.argmax(-1)
        logits = logits / temperature
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = sorted_probs.cumsum(dim=-1)
            remove = cumulative > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            sorted_probs = sorted_probs.masked_fill(remove, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)
            sampled = torch.multinomial(sorted_probs, num_samples=1).squeeze(-1)
            return sorted_idx.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: list[list[int]],
        max_gen: int,
        stop_ids: set[int],
        stop_sequences: list[list[int]] | None = None,
        temperature: float = 0.6,
        top_p: float = 0.95,
    ) -> list[list[int]]:
        dev, B = self.device, len(prompts)
        lens = [len(p) for p in prompts]
        lmax = max(lens)

        # left-pad: pad token 0 is masked at prefill and only fed to finished lanes.
        ids = torch.zeros((B, lmax), dtype=torch.long, device=dev)
        mask = torch.zeros((B, lmax), dtype=torch.long, device=dev)
        for i, p in enumerate(prompts):
            ids[i, lmax - lens[i] :] = torch.tensor(p, device=dev)
            mask[i, lmax - lens[i] :] = 1
        pos = (mask.cumsum(1) - 1).clamp(min=0)

        cache = DynamicCache()
        hidden = self.lm.model(
            inputs_embeds=self.embed(ids),
            attention_mask=mask,
            position_ids=pos,
            past_key_values=cache,
            use_cache=True,
        ).last_hidden_state[:, -1]
        nextpos = torch.tensor(lens, device=dev)

        gen: list[list[int]] = [[] for _ in range(B)]
        pending: list[list[int | None]] = [[] for _ in range(B)]
        base = [hidden[i : i + 1] for i in range(B)]  # per-lane hidden for latent proj
        done = [False] * B
        stop_sequences = stop_sequences or []

        def take(i: int, token: int) -> None:
            gen[i].append(token)
            hit_stop_seq = any(self._endswith(gen[i], seq) for seq in stop_sequences)
            if token in stop_ids or hit_stop_seq or len(gen[i]) >= max_gen:
                done[i] = True
            else:
                pending[i] = self._schedule(token)

        for i, t in enumerate(
            self._sample_next(self.lm.lm_head(hidden), temperature, top_p).tolist()
        ):
            take(i, t)

        while not all(done):
            feeds = torch.zeros(B, dtype=torch.long, device=dev)
            latent, sample = [], []
            for i in range(B):
                if done[i]:
                    continue
                f = pending[i].pop(0)
                latent.append(i) if f is None else feeds.__setitem__(i, f)
                if not pending[i]:
                    sample.append(i)
            emb = self.embed(feeds).unsqueeze(1)
            for i in latent:
                emb[i, 0] = self.proj(base[i].to(self.dtype)).view(-1)

            mask = torch.cat([mask, torch.ones(B, 1, dtype=torch.long, device=dev)], 1)
            hidden = self.lm.model(
                inputs_embeds=emb,
                attention_mask=mask,
                position_ids=nextpos.unsqueeze(1),
                past_key_values=cache,
                use_cache=True,
            ).last_hidden_state[:, -1]
            nextpos = nextpos + 1
            base = [hidden[i : i + 1] for i in range(B)]

            if sample:
                logits = self.lm.lm_head(hidden[sample])
                toks = self._sample_next(logits, temperature, top_p).tolist()
                for i, t in zip(sample, toks):
                    take(i, t)
        return gen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter_dir",
        required=True,
        help="CODI training output_dir (LoRA adapter + thought_projector.pt)",
    )
    parser.add_argument("--base_model", default="model_weights/cwm_hf")
    parser.add_argument(
        "--latent_steps", type=int, default=2, help="Must match training"
    )
    parser.add_argument("--dump_dir", default="eval-codi-cruxeval")
    parser.add_argument("--mode", default="trace_full", choices=_MODES)
    parser.add_argument("--n_samples", type=int, default=-1)
    parser.add_argument("--n_generations", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_source", nargs="+", required=True, help="dataset name(s) to merge")
    parser.add_argument("--max_gen", type=int, default=0, help="0 = mode default")
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Lanes per lock-step forward."
    )
    parser.add_argument(
        "--dist_timeout_minutes",
        type=int,
        default=360,
        help="Process-group timeout; long CODI generations can make ranks straggle.",
    )
    parser.add_argument(
        "--device_map",
        default="auto",
        help="Only used in single-process mode; ignored under torchrun.",
    )
    parser.add_argument(
        "--load_4bit",
        action="store_true",
        help="NF4 4-bit base (~16GB, fits 40GB thin); matches QLoRA-trained base.",
    )
    args = parser.parse_args()
    if args.max_gen == 0:
        args.max_gen = _MAX_GEN[args.mode]

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # --- Distributed (data-parallel) setup -------------------------------
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    ddp = world_size > 1
    if ddp:
        dist.init_process_group(
            backend="gloo",
            timeout=timedelta(minutes=args.dist_timeout_minutes),
        )
    is_rank_zero = rank == 0
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tok = AutoTokenizer.from_pretrained(args.base_model)
    eos_id = tok.eos_token_id
    eos_ids = eos_id if isinstance(eos_id, list) else [eos_id]
    stop_ids = {int(t) for t in eos_ids if t is not None}
    stop_sequences: list[list[int]] = []
    eot = tok.convert_tokens_to_ids("<|end_of_text|>")
    if eot is not None and eot != tok.unk_token_id:
        stop_ids.add(int(eot))
    if args.mode in ("direct", "reasoning"):
        stop_sequences.append(tok.encode("[/ANSWER]", add_special_tokens=False))
    elif args.mode == "trace_single_step":
        stop_ids.add(_token_id(tok, "<|frame_sep|>"))
    extract_fn = _EXTRACTORS[args.mode]

    load_kwargs: dict = {"dtype": torch.bfloat16, "attn_implementation": "sdpa"}
    if args.load_4bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    if ddp:
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        load_kwargs["device_map"] = {"": local_rank}
        base = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)
    else:
        if args.device_map not in (None, "none", "None"):
            load_kwargs["device_map"] = args.device_map
        base = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)
        device = base.device

    student = PeftModel.from_pretrained(base, args.adapter_dir)
    student.eval()

    hidden = student.config.hidden_size
    projector = build_thought_projector(hidden, device, torch.bfloat16)
    proj_path = Path(args.adapter_dir) / "thought_projector.pt"
    projector.load_state_dict(torch.load(proj_path, map_location=device))
    projector.eval()

    cfg = default_codi_config_from_tokenizer(tok, latent_steps=args.latent_steps)
    generator = CodiGenerator(student, projector, cfg, device)

    dataset = load_rows(args.data_source)
    if args.n_samples > 0:
        dataset = dataset[: args.n_samples]

    dump_path = Path(args.dump_dir)
    if is_rank_zero:
        dump_path.mkdir(parents=True, exist_ok=True)

    # Build the global work list once on rank 0, then broadcast it so every rank
    # agrees on the batch index -> samples mapping handed out by the shared counter.
    if is_rank_zero:
        kept = [
            (len(prompt), s, prompt)
            for s in dataset
            for prompt in [build_prompt_ids(args.mode, s["code"], s["input"], tok)]
        ]
        batches = make_batches(kept, args.batch_size)
    else:
        batches = None
    if ddp:
        payload = [batches]
        dist.broadcast_object_list(payload, src=0)
        batches = payload[0]
    logger.info(
        "rank %d/%d: shared queue of %d batches", rank, world_size, len(batches)
    )

    # Shared atomic work queue: a counter file in the (shared) dump dir hands out
    # the next unclaimed batch index. Idle ranks pull the next batch instead of
    # running a fixed pre-assigned slice, so a rank that draws short samples keeps
    # grabbing more rather than idling at the barrier.
    counter_path = dump_path / "_batch_counter"
    if is_rank_zero:
        counter_path.write_text("0")
    if ddp:
        dist.barrier()

    rank_results_path = dump_path / f"results_dp{rank}.jsonl"
    if rank_results_path.exists():
        rank_results_path.unlink()

    with rank_results_path.open("a") as out:

        def write(row: dict) -> None:
            out.write(json.dumps(row) + "\n")
            out.flush()
            os.fsync(out.fileno())

        n_done = 0
        while True:
            # Atomically read-and-increment the shared counter to claim a batch.
            with counter_path.open("r+") as cf:
                fcntl.flock(cf, fcntl.LOCK_EX)
                b = int(cf.read() or "0")
                if b >= len(batches):
                    break  # closing cf releases the lock
                cf.seek(0)
                cf.truncate()
                cf.write(str(b + 1))
            # b + 1 batches have now been dispatched across all ranks -> total progress.
            print(f"[dispatch {b + 1}/{len(batches)}] rank {rank}", flush=True)
            batch = batches[b]
            prompts = [prompt for _, _, prompt in batch]
            t0 = time.perf_counter()
            sample_gens: list[list[dict]] = [[] for _ in batch]
            for _ in range(args.n_generations):
                gens = generator.generate_batch(
                    prompts,
                    args.max_gen,
                    stop_ids,
                    stop_sequences,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                for i, ((_sort_len, s, _prompt), g) in enumerate(zip(batch, gens)):
                    generation = tok.decode(g, skip_special_tokens=False)
                    predicted = extract_fn(generation, s["input"])
                    correct = (
                        check_correct(s["code"], s["output"], predicted)
                        if predicted is not None
                        else False
                    )
                    sample_gens[i].append(
                        {
                            "generation": generation,
                            "predicted": predicted,
                            "correct": correct,
                        }
                    )
            dt = (time.perf_counter() - t0) / (len(batch) * args.n_generations)
            for (_sort_len, s, _prompt), gens_for_sample in zip(batch, sample_gens):
                n_correct = sum(g["correct"] for g in gens_for_sample)
                write(
                    {
                        "id": s["id"],
                        "mode": args.mode,
                        "expected": s["output"],
                        "pass_at_1": n_correct / args.n_generations,
                        "generations": gens_for_sample,
                    }
                )
            n_done += len(batch)
            logger.info(
                "rank %d claimed batch %d/%d (%d samples done) %.1fs/generation",
                rank,
                b,
                len(batches),
                n_done,
                dt,
            )
    logger.info("rank %d: %d samples scored across claimed batches", rank, n_done)

    if ddp:
        dist.barrier()

    if is_rank_zero:
        all_results: list[dict] = []
        for r in range(world_size):
            rank_path = dump_path / f"results_dp{r}.jsonl"
            with rank_path.open("r") as f:
                all_results.extend(json.loads(line) for line in f)

        id_to_idx = {s["id"]: i for i, s in enumerate(dataset)}
        all_results.sort(key=lambda r: id_to_idx.get(r["id"], 0))

        pass_at_1 = (
            sum(r["pass_at_1"] for r in all_results) / len(all_results)
            if all_results
            else 0.0
        )
        print(
            f"\nCRUXEval-O {args.mode} pass@1: {pass_at_1:.4f} "
            f"(n={len(all_results)}, {args.n_generations} gens each)"
        )

        with (dump_path / "results.jsonl").open("w") as f:
            for r in all_results:
                f.write(json.dumps(r) + "\n")

        summary = {
            "pass_at_1": pass_at_1,
            "n_total": len(all_results),
            "max_gen": args.max_gen,
            "batch_size": args.batch_size,
            "n_generations": args.n_generations,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "use_sampling": args.temperature > 0,
            "seed": args.seed,
            "mode": args.mode,
            "base_model": args.base_model,
            "adapter_dir": args.adapter_dir,
            "latent_steps": args.latent_steps,
            "batching": "round_robin",
            "loader": "huggingface+codi",
        }
        with (dump_path / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2)
        counter_path.unlink(missing_ok=True)
        logger.info("Results written to %s", dump_path)

    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
