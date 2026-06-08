# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
CRUXEval-O output prediction for a CODI-distilled CWM model.

Predicts the *execution result* (CRUXEval-O pass@1), not the trace-quality
metrics of ``evals.trace_analysis``. The CODI student was distilled on execution
traces (``cwm.training.data`` uses the full-trace prompt), so it predicts the
output by generating a trace with latent reasoning and reading the return value
from the final RETURN frame (``extract_answer_trace_full``).

Loads a HuggingFace CWM base + a trained CODI LoRA adapter + ``thought_projector.pt``
(``cwm.training.train`` output). Whenever the model emits ``<|line_sep|>`` it
injects ``latent_steps`` continuous thoughts in place of the per-frame locals,
mirroring the no-grad path of ``cwm.training.codi_streaming``. Greedy decoding,
pass@1 (matches the greedy Table 9 reproduction config).

    torchrun --nproc_per_node=8 -m evals.cruxeval.run_eval_codi \\
        --adapter_dir /path/to/codi_qlora_..._output_dir \\
        --base_model model_weights/cwm_hf \\
        --latent_steps 2 \\
        --dump_dir eval-codi-cruxeval
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch import nn
from tqdm import tqdm
from transformers.cache_utils import DynamicCache

from cwm.training.codi_config import default_codi_config_from_tokenizer
from cwm.training.data import cruxeval_split
from evals.cruxeval.evaluate import check_correct, extract_answer_trace_full
from evals.cruxeval.prompts import _make_trace_context

logger = logging.getLogger(__name__)

# Full traces can be long; 8192 matches the cruxeval trace_full budget.
_MAX_GEN = 8192


def build_trace_full_prompt_ids(code: str, input_str: str, tok) -> list[int]:
    """HF-tokenizer port of ``make_trace_full_prompt_tokens`` (the seeded
    full-trace prompt the CODI student was trained on):
    ``[BOS][TRACE_CONTEXT_START]$CONTEXT[FRAME_SEP][CALL_SEP]{}
    [ACTION_SEP]def main():\\n[FRAME_SEP]``.
    """
    sid = tok.convert_tokens_to_ids
    context = _make_trace_context(code, input_str)
    ids = [tok.bos_token_id, sid("<|trace_context_start|>")]
    ids += tok.encode(context, add_special_tokens=False)
    ids += [sid("<|frame_sep|>"), sid("<|call_sep|>")]
    ids += tok.encode("{}", add_special_tokens=False)
    ids += [sid("<|action_sep|>")]
    ids += tok.encode("def main():\n", add_special_tokens=False)
    ids += [sid("<|frame_sep|>")]
    return ids


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
    """Greedy autoregressive generation with CODI latent injection.

    Mirrors the no-grad path of ``cwm.training.codi_streaming`` (``_latent_step`` /
    ``_latent_block_plain``): each model call appends one position to the KV cache;
    on ``<|line_sep|>`` the per-frame locals are replaced by an optional
    latent-start token, ``latent_steps`` projected continuous thoughts, and an
    optional latent-end token. The next visible token (normally ``<|action_sep|>``)
    is then decoded from the latent block's final logits.
    """

    def __init__(self, student, projector, cfg, device) -> None:
        # student.base_model.model is the wrapped CwmForCausalLM (see codi_streaming).
        self.causal_lm = student.base_model.model
        self.embed = student.get_input_embeddings()
        self.proj = projector
        self.cfg = cfg
        self.device = device
        self.dtype = self.embed.weight.dtype
        self.hidden = self.embed.weight.shape[-1]

    def _token_embed(self, token_id: int) -> torch.Tensor:
        ids = torch.tensor([[token_id]], device=self.device)
        return self.embed(ids)  # [1, 1, hidden]

    def _forward(self, inputs_embeds, cur_len, cache):
        """One model call. Returns (last_hidden[1,L,h], next_logits[vocab], new_len)."""
        L = inputs_embeds.shape[1]
        pos = torch.arange(cur_len, cur_len + L, device=self.device).unsqueeze(0)
        attn = torch.ones(1, cur_len + L, device=self.device, dtype=torch.long)
        out = self.causal_lm.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            position_ids=pos,
            past_key_values=cache,
            use_cache=True,
        )
        last_hidden = out.last_hidden_state
        logits = self.causal_lm.lm_head(last_hidden[:, -1])[0]  # [vocab]
        return last_hidden, logits, cur_len + L

    def _latent_input(self, base: torch.Tensor) -> torch.Tensor:
        """Project the previous step's last hidden state into an embedding slot."""
        latent = self.proj(base.to(device=self.device, dtype=self.dtype))
        return latent.view(1, 1, -1)

    @torch.no_grad()
    def generate(self, prompt_ids: list[int], max_gen: int, stop_ids: set[int]) -> list[int]:
        cfg = self.cfg
        cache = DynamicCache()
        prompt_embeds = self.embed(torch.tensor([prompt_ids], device=self.device))
        _, logits, cur = self._forward(prompt_embeds, 0, cache)

        generated: list[int] = []
        for _ in range(max_gen):
            token = int(logits.argmax(-1))
            generated.append(token)
            if token in stop_ids:
                break

            if token == cfg.latent_span_start_token_id:
                # Feed the line_sep token itself, then inject the latent block in
                # place of the per-frame locals.
                _, _, cur = self._forward(self._token_embed(token), cur, cache)
                base = None
                if cfg.latent_start_token_id is not None:
                    lh, _, cur = self._forward(
                        self._token_embed(cfg.latent_start_token_id), cur, cache
                    )
                    base = lh[:, -1]
                for _ in range(cfg.latent_steps):
                    if base is None:  # no latent-start token: seed from zeros
                        base = torch.zeros(1, self.hidden, device=self.device, dtype=self.dtype)
                    lh, logits, cur = self._forward(self._latent_input(base), cur, cache)
                    base = lh[:, -1]
                if cfg.latent_end_token_id is not None:
                    _, logits, cur = self._forward(
                        self._token_embed(cfg.latent_end_token_id), cur, cache
                    )
                # `logits` now predicts the next visible token (normally action_sep).
            else:
                _, logits, cur = self._forward(self._token_embed(token), cur, cache)

        return generated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_dir", required=True, help="CODI training output_dir (LoRA adapter + thought_projector.pt)")
    parser.add_argument("--base_model", default="model_weights/cwm_hf")
    parser.add_argument("--latent_steps", type=int, default=2, help="Must match training")
    parser.add_argument("--dump_dir", default="eval-codi-cruxeval")
    parser.add_argument("--n_samples", type=int, default=-1)
    parser.add_argument(
        "--data_split",
        default="val",
        choices=["train", "val", "all"],
        help="cruxeval_split: sweep on held-out 'val' (matches train data_split=train)",
    )
    parser.add_argument("--max_gen", type=int, default=_MAX_GEN)
    parser.add_argument(
        "--device_map",
        default="auto",
        help="Only used in single-process mode; ignored under torchrun.",
    )
    args = parser.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # --- Distributed (data-parallel) setup -------------------------------
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    ddp = world_size > 1
    if ddp:
        dist.init_process_group(backend="gloo")
    is_rank_zero = rank == 0

    tok = AutoTokenizer.from_pretrained(args.base_model)
    eos_id = tok.eos_token_id
    eos_ids = eos_id if isinstance(eos_id, list) else [eos_id]
    stop_ids = set(eos_ids)
    eot = tok.convert_tokens_to_ids("<|end_of_text|>")
    if eot is not None and eot != tok.unk_token_id:
        stop_ids.add(int(eot))

    load_kwargs: dict = {"dtype": torch.bfloat16, "attn_implementation": "sdpa"}
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

    dataset = cruxeval_split(load_dataset("cruxeval-org/cruxeval", split="test"), args.data_split)
    if args.n_samples > 0:
        dataset = dataset[: args.n_samples]
    my_samples = dataset[rank::world_size] if ddp else dataset
    logger.info("rank %d/%d: %d samples", rank, world_size, len(my_samples))

    dump_path = Path(args.dump_dir)
    if is_rank_zero:
        dump_path.mkdir(parents=True, exist_ok=True)
    if ddp:
        dist.barrier()

    results: list[dict] = []
    pbar = tqdm(total=len(my_samples), desc=f"CODI-CruxEval [rank={rank}]", position=rank)
    for sample in my_samples:
        code = sample["code"]
        inp = sample["input"]
        expected = sample["output"]

        prompt_ids = build_trace_full_prompt_ids(code, inp, tok)
        gen_ids = generator.generate(prompt_ids, args.max_gen, stop_ids)
        generation = tok.decode(gen_ids, skip_special_tokens=False)

        predicted = extract_answer_trace_full(generation, inp)
        correct = (
            check_correct(code, expected, predicted) if predicted is not None else False
        )

        results.append(
            {
                "id": sample["id"],
                "code": code,
                "input": inp,
                "expected": expected,
                "predicted": predicted,
                "correct": correct,
                "generation": generation,
            }
        )
        pbar.update(1)
    pbar.close()

    rank_results_path = dump_path / f"results_dp{rank}.jsonl"
    with rank_results_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

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

        pass_at_1 = sum(1.0 for r in all_results if r["correct"]) / len(all_results)
        print(f"\nCRUXEval-O pass@1: {pass_at_1:.4f} (n={len(all_results)}, greedy)")

        with (dump_path / "results.jsonl").open("w") as f:
            for r in all_results:
                f.write(json.dumps(r) + "\n")

        summary = {
            "pass_at_1": pass_at_1,
            "n_total": len(all_results),
            "decoding": "greedy",
            "mode": "trace_full",
            "base_model": args.base_model,
            "adapter_dir": args.adapter_dir,
            "latent_steps": args.latent_steps,
            "data_split": args.data_split,
            "loader": "huggingface+codi",
        }
        with (dump_path / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Results written to %s", dump_path)

    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
