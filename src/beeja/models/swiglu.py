"""SwiGLU feed-forward network.

A gated variant of the MLP. Instead of one hidden projection + activation, it
uses two parallel projections — a *gate* and an *up* — combines them with a SiLU
(swish) gate, then projects down:

    SwiGLU(x) = ( SiLU(x W_gate) ⊙ (x W_up) ) W_down
    SiLU(z)   = z · sigmoid(z)

The gating lets the network modulate information multiplicatively, which tends to
outperform a plain GELU MLP at equal cost. To keep the parameter budget close to
the 4× GELU MLP (which is 8·d²), the hidden size is ~8d/3 across three matrices.

    x shape: [B, T, d]  ->  [B, T, d]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from beeja.models.config import ModelConfig


def _round_to(n: int, multiple: int) -> int:
    return ((n + multiple - 1) // multiple) * multiple


def swiglu_hidden(n_embd: int) -> int:
    """Hidden width chosen so total params ≈ a 4× GELU MLP (8·d²)."""
    return _round_to(8 * n_embd // 3, 8)


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig, hidden: int | None = None) -> None:
        super().__init__()
        d = config.n_embd
        hidden = swiglu_hidden(d) if hidden is None else hidden
        self.w_gate = nn.Linear(d, hidden, bias=config.bias)
        self.w_up = nn.Linear(d, hidden, bias=config.bias)
        self.w_down = nn.Linear(hidden, d, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))
