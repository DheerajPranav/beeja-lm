"""A single Transformer block (pre-normalization).

Design decision: **pre-norm** residual connections, i.e. normalize *before* each
sub-layer and add the result back:

    x = x + Attention(LayerNorm(x))
    x = x + MLP(LayerNorm(x))

Pre-norm keeps a clean identity path from input to output, which makes deep
stacks train stably without a learning-rate warmup as delicate as post-norm
needs. We keep this consistent across every block.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from beeja.models.attention import MultiHeadSelfAttention
from beeja.models.config import ModelConfig
from beeja.models.mlp import MLP


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = MultiHeadSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
