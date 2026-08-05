"""A single Transformer block (pre-normalization), with pluggable norm and MLP.

Design decision: **pre-norm** residual connections, i.e. normalize *before* each
sub-layer and add the result back:

    x = x + Attention(Norm(x))
    x = x + FeedForward(Norm(x))

Pre-norm keeps a clean identity path from input to output, which makes deep
stacks train stably. The normalization (LayerNorm / RMSNorm) and the feed-forward
(GELU MLP / SwiGLU) are selected from the model config, so a "baseline" and a
"modern" block share this exact structure and differ only in those swappable parts.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from beeja.models.attention import MultiHeadSelfAttention
from beeja.models.config import ModelConfig
from beeja.models.mlp import MLP
from beeja.models.rmsnorm import RMSNorm
from beeja.models.swiglu import SwiGLU


def make_norm(config: ModelConfig) -> nn.Module:
    if config.norm == "rmsnorm":
        return RMSNorm(config.n_embd)
    return nn.LayerNorm(config.n_embd, bias=config.bias)


def make_mlp(config: ModelConfig) -> nn.Module:
    if config.mlp == "swiglu":
        return SwiGLU(config)
    return MLP(config)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln1 = make_norm(config)
        self.attn = MultiHeadSelfAttention(config)
        self.ln2 = make_norm(config)
        self.mlp = make_mlp(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

    def forward_cached(
        self, x: torch.Tensor, past_kv: tuple[torch.Tensor, torch.Tensor] | None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Pre-norm block step that threads a KV cache. Returns (x, present_kv)."""
        attn_out, present = self.attn(self.ln1(x), past_kv=past_kv, use_cache=True)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, present
