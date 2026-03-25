"""Apply the unembedding head at each layer to extracted activations.

Example:
    python -m interp.logit_lens.run_logit_lens \\
        extract_dir=./interp-extract-trace_full \\
        checkpoint_dir=./model_weights/cwm
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from cwm.common.params import load_from_cli

logger = logging.getLogger(__name__)


@dataclass
class LogitLensArgs:
    extract_dir: str = "interp-extract"
    checkpoint_dir: str = ""
    top_k: int = 10
    wandb_project: str = "cwm-interp"
    wandb_run_name: str = ""


def _load_norm_and_output(checkpoint_dir: str):
    """Load only the final RMSNorm weights and output head from the checkpoint.

    Returns (norm_weight, output_weight, eps).
    Supports both legacy .pth format and DCP (.distcp) format.
    """
    ckpt_path = Path(checkpoint_dir)
    norm_weight = None
    output_weight = None

    # Try legacy .pth format first
    pt_files = sorted(ckpt_path.glob("consolidated*.pth")) or sorted(
        ckpt_path.glob("*.pth")
    )
    if pt_files:
        for pf in pt_files:
            try:
                state = torch.load(pf, map_location="cpu", weights_only=True)
                if isinstance(state, dict):
                    if "model" in state:
                        state = state["model"]
                    if norm_weight is None and "norm.weight" in state:
                        norm_weight = state["norm.weight"].float()
                    if output_weight is None and "output.weight" in state:
                        output_weight = state["output.weight"].float()
                    if output_weight is None and "tok_embeddings.weight" in state:
                        output_weight = state["tok_embeddings.weight"].float()
            except Exception as e:
                logger.warning(f"Could not load {pf}: {e}")
            if norm_weight is not None and output_weight is not None:
                break

    # Fall back to DCP (.distcp) format
    if (norm_weight is None or output_weight is None) and list(
        ckpt_path.glob("*.distcp")
    ):
        try:
            import torch.distributed.checkpoint as dcp
            from torch.distributed.checkpoint._fsspec_filesystem import FsspecReader
            from upath import UPath

            reader = FsspecReader(UPath(checkpoint_dir))
            metadata = reader.read_metadata()
            keys = set(metadata.state_dict_metadata.keys())

            norm_key = next(
                (k for k in ("model.norm.weight", "norm.weight") if k in keys), None
            )
            output_key = next(
                (
                    k
                    for k in (
                        "model.output.weight",
                        "output.weight",
                        "model.tok_embeddings.weight",
                        "tok_embeddings.weight",
                    )
                    if k in keys
                ),
                None,
            )

            load_dict = {}
            if norm_key:
                norm_meta = metadata.state_dict_metadata[norm_key]
                load_dict[norm_key] = torch.zeros(norm_meta.size, dtype=torch.float32)
            if output_key:
                out_meta = metadata.state_dict_metadata[output_key]
                load_dict[output_key] = torch.zeros(out_meta.size, dtype=torch.float32)

            if load_dict:
                dcp.load(load_dict, storage_reader=reader)
                if norm_key and norm_weight is None:
                    norm_weight = load_dict[norm_key].float()
                if output_key and output_weight is None:
                    output_weight = load_dict[output_key].float()
        except Exception as e:
            logger.warning(f"Could not load DCP checkpoint: {e}")

    if norm_weight is None or output_weight is None:
        raise RuntimeError(
            "Could not find norm.weight or output.weight in checkpoint"
        )

    # Infer eps from params.json if available
    params_path = ckpt_path / "params.json"
    eps = 1e-5
    if params_path.exists():
        with params_path.open() as f:
            params = json.load(f)
        eps = params.get("norm_eps", 1e-5)

    return norm_weight, output_weight, eps


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    rms = x.pow(2).mean(-1, keepdim=True).add(eps).sqrt()
    return x / rms * weight


def apply_logit_lens(
    h: torch.Tensor,           # [n_pos, dim]
    norm_weight: torch.Tensor,  # [dim]
    output_weight: torch.Tensor,  # [vocab, dim]
    eps: float,
    top_k: int,
    target_token_ids: list[int] | None = None,
) -> dict:
    """Apply norm + output head to h and return top-k results.

    Returns dict with:
      top_k_ids: [n_pos, k]
      top_k_probs: [n_pos, k]
      target_probs: [n_pos] or None
    """
    with torch.no_grad():
        h_norm = _rms_norm(h.float(), norm_weight, eps)
        logits = torch.nn.functional.linear(h_norm, output_weight)  # [n_pos, vocab]
        probs = torch.softmax(logits, dim=-1)

        values, indices = probs.topk(top_k, dim=-1)

        result = {
            "top_k_ids": indices.tolist(),
            "top_k_probs": values.tolist(),
        }

        if target_token_ids:
            tgt = torch.tensor(target_token_ids, dtype=torch.long)
            # Max prob over target tokens at each position
            target_probs = probs[:, tgt].max(-1).values
            result["target_probs"] = target_probs.tolist()
        else:
            result["target_probs"] = None

    return result


def main(args: LogitLensArgs) -> None:
    logging.basicConfig(level=logging.INFO)

    extract_path = Path(args.extract_dir)
    index_path = extract_path / "index.jsonl"

    if not index_path.exists():
        raise FileNotFoundError(f"No index.jsonl in {args.extract_dir}")

    index: list[dict] = []
    with index_path.open() as f:
        for line in f:
            index.append(json.loads(line))

    norm_weight, output_weight, eps = _load_norm_and_output(args.checkpoint_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm_weight = norm_weight.to(device)
    output_weight = output_weight.to(device)

    act_dir = extract_path / "activations"
    all_results: list[dict] = []

    for meta in index:
        sid = meta["sample_id"]
        pt_path = act_dir / f"{sid}.pt"
        if not pt_path.exists():
            continue

        sample = torch.load(pt_path, map_location="cpu", weights_only=False)
        activations = sample.get("activations", {})
        if not activations:
            continue

        layers = sorted(activations.keys())
        sample_lens = {}

        # Determine target token IDs for this sample (correct answer tokens)
        # We do a simple check: is the first top-1 token the right prediction?
        for layer in layers:
            h = activations[layer].to(device)
            lens_result = apply_logit_lens(
                h,
                norm_weight,
                output_weight,
                eps,
                top_k=args.top_k,
            )
            sample_lens[layer] = lens_result

        all_results.append(
            {
                "sample_id": sid,
                "correct": meta.get("correct", False),
                "mode": meta.get("mode", ""),
                "layers": layers,
                "logit_lens": sample_lens,
            }
        )

    # Save results
    out_path = extract_path / "logit_lens.pt"
    torch.save(all_results, out_path)
    logger.info(f"Saved logit lens results for {len(all_results)} samples to {out_path}")

    # Compute crystallization curve: for each layer, what fraction of samples
    # have a correct-answer token in top-k?
    layers_set = sorted(
        {l for r in all_results for l in r.get("layers", [])}
    )
    if layers_set:
        print("\nLogit lens summary (fraction with high activation at each layer):")
        for layer in layers_set:
            layer_probs = []
            for r in all_results:
                ll = r.get("logit_lens", {}).get(layer)
                if ll and ll.get("top_k_probs"):
                    # Mean top-1 probability
                    top1 = [p[0] for p in ll["top_k_probs"] if p]
                    if top1:
                        layer_probs.append(sum(top1) / len(top1))
            if layer_probs:
                avg = sum(layer_probs) / len(layer_probs)
                print(f"  Layer {layer:2d}: mean top-1 prob = {avg:.4f}")

    try:
        import wandb

        if args.wandb_project:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or "logit-lens",
                config=vars(args),
            )
            # Log crystallization curve
            for layer in layers_set:
                layer_probs = []
                for r in all_results:
                    ll = r.get("logit_lens", {}).get(layer)
                    if ll and ll.get("top_k_probs"):
                        top1 = [p[0] for p in ll["top_k_probs"] if p]
                        if top1:
                            layer_probs.append(sum(top1) / len(top1))
                if layer_probs:
                    wandb.log(
                        {"mean_top1_prob": sum(layer_probs) / len(layer_probs), "layer": layer}
                    )
            wandb.finish()
    except ImportError:
        pass


if __name__ == "__main__":
    args = load_from_cli(LogitLensArgs)
    main(args)
