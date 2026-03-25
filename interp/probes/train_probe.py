"""Train linear/MLP probes on extracted activations.

Example:
    python -m interp.probes.train_probe \\
        extract_dir=./interp-extract-trace_full \\
        target_property=will_be_correct \\
        probe_type=linear
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split

from cwm.common.params import load_from_cli
from interp.probes.dataset import ProbeDataset
from interp.probes.models import LinearProbe, MLPProbe

logger = logging.getLogger(__name__)


@dataclass
class ProbeTrainArgs:
    extract_dir: str = "interp-extract"
    target_property: str = "will_be_correct"
    probe_type: str = "linear"  # linear | mlp1 | mlp2
    layers: list[int] = field(
        default_factory=lambda: [0, 8, 16, 24, 32, 40, 48, 56, 63]
    )
    hidden_dim: int = 256
    lr: float = 1e-3
    epochs: int = 50
    batch_size: int = 256
    val_fraction: float = 0.2
    position_filter: str = "all"
    wandb_project: str = "cwm-interp"
    wandb_run_name: str = ""


def build_probe(probe_type: str, in_dim: int, n_classes: int, hidden_dim: int):
    if probe_type == "linear":
        return LinearProbe(in_dim, n_classes)
    elif probe_type == "mlp1":
        return MLPProbe(in_dim, hidden_dim, n_classes, n_hidden=1)
    elif probe_type == "mlp2":
        return MLPProbe(in_dim, hidden_dim, n_classes, n_hidden=2)
    else:
        raise ValueError(f"Unknown probe_type: {probe_type!r}")


def train_one_layer(
    dataset: ProbeDataset,
    probe_type: str,
    hidden_dim: int,
    lr: float,
    epochs: int,
    batch_size: int,
    val_fraction: float,
    device: torch.device,
) -> dict:
    n = len(dataset)
    if n == 0:
        return {}

    n_val = max(1, int(n * val_fraction))
    n_train = n - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    probe = build_probe(probe_type, dataset.dim, dataset.n_classes, hidden_dim).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Random baseline
    labels_all = [dataset[i][1] for i in range(n)]
    from collections import Counter

    cnt = Counter(labels_all)
    majority_acc = cnt.most_common(1)[0][1] / n

    history = []
    for epoch in range(epochs):
        probe.train()
        total_loss, total_correct, total_n = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device=device, dtype=torch.float32), y.to(device)
            logits = probe(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            total_correct += (logits.argmax(-1) == y).sum().item()
            total_n += len(y)

        probe.eval()
        val_correct, val_n = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device=device, dtype=torch.float32), y.to(device)
                logits = probe(x)
                val_correct += (logits.argmax(-1) == y).sum().item()
                val_n += len(y)

        train_acc = total_correct / total_n if total_n > 0 else 0.0
        val_acc = val_correct / val_n if val_n > 0 else 0.0
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / total_n if total_n > 0 else 0.0,
                "train_acc": train_acc,
                "val_acc": val_acc,
            }
        )

    best_val_acc = max(h["val_acc"] for h in history) if history else 0.0
    return {
        "best_val_acc": best_val_acc,
        "majority_baseline": majority_acc,
        "history": history,
    }


def main(args: ProbeTrainArgs) -> None:
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        import wandb

        use_wandb = bool(args.wandb_project)
        if use_wandb:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name
                or f"probe-{args.target_property}-{args.probe_type}",
                config=vars(args),
            )
    except ImportError:
        use_wandb = False

    layer_results = {}
    for layer in args.layers:
        logger.info(
            f"Training {args.probe_type} probe for {args.target_property} at layer {layer}"
        )
        try:
            ds = ProbeDataset(
                extract_dir=args.extract_dir,
                layer=layer,
                target_property=args.target_property,
                position_filter=args.position_filter,
            )
        except Exception as e:
            logger.warning(f"Layer {layer}: failed to load dataset: {e}")
            continue

        if len(ds) == 0:
            logger.warning(f"Layer {layer}: empty dataset, skipping")
            continue

        result = train_one_layer(
            dataset=ds,
            probe_type=args.probe_type,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            val_fraction=args.val_fraction,
            device=device,
        )
        layer_results[layer] = result
        logger.info(
            f"  Layer {layer}: best_val_acc={result.get('best_val_acc', 0):.4f}, "
            f"majority={result.get('majority_baseline', 0):.4f}"
        )

        if use_wandb:
            wandb.log(
                {
                    f"layer_{layer}/best_val_acc": result.get("best_val_acc", 0),
                    f"layer_{layer}/majority_baseline": result.get(
                        "majority_baseline", 0
                    ),
                    "layer": layer,
                }
            )
            for h in result.get("history", []):
                wandb.log(
                    {
                        f"layer_{layer}/train_acc": h["train_acc"],
                        f"layer_{layer}/val_acc": h["val_acc"],
                        f"layer_{layer}/train_loss": h["train_loss"],
                        "epoch": h["epoch"],
                    }
                )

    if use_wandb:
        import wandb as _wandb

        # Summary table: accuracy vs layer
        layer_accs = [
            (l, r.get("best_val_acc", 0)) for l, r in sorted(layer_results.items())
        ]
        if layer_accs:
            table = _wandb.Table(
                columns=["layer", "val_acc"],
                data=[[l, a] for l, a in layer_accs],
            )
            _wandb.log({"accuracy_by_layer": table})
        _wandb.finish()


if __name__ == "__main__":
    args = load_from_cli(ProbeTrainArgs)
    main(args)
