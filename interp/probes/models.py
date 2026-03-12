"""Probe model architectures."""

import torch
import torch.nn as nn


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, n_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MLPProbe(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        n_classes: int,
        n_hidden: int = 1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.ReLU())
            prev = hidden_dim
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
