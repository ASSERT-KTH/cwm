"""Plotting utilities for wandb logging."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import wandb


def plot_layer_heatmap(
    data: list[list[float]],
    xlabel: str = "Position",
    ylabel: str = "Layer",
    title: str = "",
) -> "wandb.Image":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import wandb

    arr = np.array(data)
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(arr, aspect="auto", origin="lower")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def plot_accuracy_by_layer(
    layer_accs: list[tuple[int, float]],
    baseline: float = 0.0,
    title: str = "Probe accuracy by layer",
) -> "wandb.plot":
    import wandb

    table = wandb.Table(
        columns=["layer", "accuracy"],
        data=[[l, a] for l, a in layer_accs],
    )
    return wandb.plot.line(table, "layer", "accuracy", title=title)


def plot_steering_sweep(
    alphas: list[float],
    pass_at_1: list[float],
    layer: int,
) -> "wandb.plot":
    import wandb

    table = wandb.Table(
        columns=["alpha", "pass_at_1"],
        data=[[a, p] for a, p in zip(alphas, pass_at_1)],
    )
    return wandb.plot.line(table, "alpha", "pass_at_1", title=f"Steering sweep layer {layer}")


def plot_crystallization(
    layer_probs: list[tuple[int, float]],
    title: str = "Logit lens: mean top-1 prob by layer",
) -> "wandb.plot":
    import wandb

    table = wandb.Table(
        columns=["layer", "mean_top1_prob"],
        data=[[l, p] for l, p in layer_probs],
    )
    return wandb.plot.line(table, "layer", "mean_top1_prob", title=title)
