"""Causal self-attention, implemented manually (no fused SDPA kernel).

Scaled dot-product attention:

    Attention(Q, K, V) = softmax( (Q Kᵀ / sqrt(d_k)) + causal_mask ) V

The causal mask sets every *future* key position to ``-inf`` before softmax, so
a position can attend only to itself and earlier positions — the property that
makes next-token prediction well-posed.

Two implementations:

- ``SingleHeadAttention``: one head, the clearest teaching reference.
- ``MultiHeadSelfAttention``: all heads at once via a single fused QKV projection
  and a reshape into ``[B, n_head, T, head_size]``. This is what the blocks use.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from beeja.models.config import ModelConfig
from beeja.models.rope import apply_rope, build_rope_cache


class SingleHeadAttention(nn.Module):
    """One causal attention head. Shapes: x [B,T,C] -> out [B,T,head_size]."""

    def __init__(
        self, n_embd: int, head_size: int, block_size: int, dropout: float = 0.0, bias: bool = True
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=bias)
        self.query = nn.Linear(n_embd, head_size, bias=bias)
        self.value = nn.Linear(n_embd, head_size, bias=bias)
        self.dropout = nn.Dropout(dropout)
        # Lower-triangular mask; buffer so it moves with .to(device) but is not a param.
        self.register_buffer(
            "mask", torch.tril(torch.ones(block_size, block_size)).view(1, block_size, block_size)
        )

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        _, t, _ = x.shape
        q = self.query(x)  # [B,T,hs]
        k = self.key(x)  # [B,T,hs]
        v = self.value(x)  # [B,T,hs]

        att = (q @ k.transpose(-2, -1)) * self.head_size**-0.5  # [B,T,T]
        att = att.masked_fill(self.mask[:, :t, :t] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)  # rows sum to 1 over allowed keys
        att = self.dropout(att)
        out = att @ v  # [B,T,hs]
        if return_attn:
            return out, att
        return out


class MultiHeadSelfAttention(nn.Module):
    """Multi-head causal self-attention. Shapes: x [B,T,C] -> out [B,T,C]."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = config.n_head
        self.head_size = config.head_size
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        # RoPE rotates Q/K by position inside attention (no learned position embedding).
        self.use_rope = config.pos_encoding == "rope"
        if self.use_rope:
            cos, sin = build_rope_cache(config.block_size, config.head_size)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ):
        """Compute attention, optionally reusing/extending a KV cache.

        With ``past_kv`` (cached keys/values for earlier positions), only the new
        tokens' K/V are computed and appended — turning per-step generation from
        O(T²) into O(T). ``past_kv`` / ``use_cache`` are unused during training.
        """
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=2)  # each [B,T,C]
        # Split channels into heads: [B,T,C] -> [B, n_head, T, head_size]
        q = q.view(b, t, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_size).transpose(1, 2)

        past_len = past_kv[0].size(2) if past_kv is not None else 0
        if self.use_rope:  # rotate new Q/K at their absolute positions
            cos = self.rope_cos[past_len : past_len + t]
            sin = self.rope_sin[past_len : past_len + t]
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        if past_kv is not None:  # prepend cached keys/values along time
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        present = (k, v)

        tk = k.size(2)
        att = (q @ k.transpose(-2, -1)) * self.head_size**-0.5  # [B,nh,t,tk]
        # New queries occupy absolute rows [past_len, past_len+t); mask future keys.
        att = att.masked_fill(self.mask[:, :, past_len : past_len + t, :tk] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # [B,nh,t,hs]
        # Recombine heads: [B,nh,t,hs] -> [B,t,C]
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        y = self.resid_dropout(self.proj(y))
        if return_attn:
            return y, att
        if use_cache:
            return y, present
        return y
