"""Reusable model heads."""

from __future__ import annotations

import torch
from torch import nn


class ConditionLevelEmbedding(nn.Module):
    def __init__(self, num_conditions: int = 5, num_levels: int = 5, dim: int = 16) -> None:
        super().__init__()
        self.condition = nn.Embedding(num_conditions, dim)
        self.level = nn.Embedding(num_levels, dim)
        self.output_dim = dim * 2

    def forward(self, condition_idx: torch.Tensor, level_idx: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.condition(condition_idx), self.level(level_idx)], dim=1)


def mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.2) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )

