"""Position-wise feed-forward network (MLP).

Applied identically at every position. Expands the model dimension by 4x, adds a
GELU non-linearity, then projects back:

    x [B,T,C] -> fc -> [B,T,4C] -> GELU -> proj -> [B,T,C]

This is where most of a Transformer's parameters and per-token computation live.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from beeja.models.config import ModelConfig


class MLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden = 4 * config.n_embd
        self.fc = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.act = nn.GELU()
        self.proj = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(self.act(self.fc(x))))
