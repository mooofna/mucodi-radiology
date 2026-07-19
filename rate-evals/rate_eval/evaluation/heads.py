"""Classifier heads for frozen-feature linear / MLP probing."""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn


class Linear(nn.Module):
    """Single linear layer. Bag input (B, T, D) is mean-pooled to (B, D)."""

    def __init__(self, dim_input: int, dim_output: int):
        super().__init__()
        self.fc = nn.Linear(dim_input, dim_output)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        if x.ndim == 3:
            x = x.mean(dim=1)
        elif x.ndim != 2:
            raise ValueError(f"Expected 2D or 3D input, got shape {tuple(x.shape)}")
        return self.fc(x)


class MLP(nn.Module):
    """Generic MLP: `num_layers` linear layers, hidden layers all width `dim_hidden`, ReLU + Dropout between."""

    def __init__(
        self,
        dim_input: int,
        dim_hidden: int,
        dim_output: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        layers: List[nn.Module] = []
        in_dim = dim_input
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, dim_hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = dim_hidden
        layers.append(nn.Linear(in_dim, dim_output))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        if x.ndim == 3:
            x = x.mean(dim=1)
        elif x.ndim != 2:
            raise ValueError(f"Expected 2D or 3D input, got shape {tuple(x.shape)}")
        return self.mlp(x)


def build_head(spec: Dict[str, Any], *, dim_input: int, dim_output: int) -> nn.Module:
    """Construct a head from a spec dict; `spec["kind"]` selects it, remaining keys become constructor kwargs."""
    kind = spec.get("kind", "linear").lower()
    if kind == "linear":
        return Linear(dim_input=dim_input, dim_output=dim_output)
    if kind == "mlp":
        return MLP(
            dim_input=dim_input,
            dim_hidden=int(spec.get("dim_hidden", 512)),
            dim_output=dim_output,
            num_layers=int(spec.get("num_layers", 2)),
            dropout=float(spec.get("dropout", 0.25)),
        )
    raise ValueError(f"Unknown head kind: {kind!r}")
