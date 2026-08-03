"""Rotary Position Embeddings (RoPE).

Instead of *adding* a learned position vector, RoPE *rotates* each query/key by
an angle proportional to its position. The dot product of a rotated query at
position m and a rotated key at position n then depends only on their relative
distance (m − n) — so the model gets relative-position awareness for free, and
it extrapolates past the trained context better than learned absolute embeddings.

For a feature pair (x_2i, x_2i+1) at position p with frequency θ_i:

    [x'_2i, x'_2i+1] = R(p·θ_i) · [x_2i, x_2i+1]

We use the "rotate-half" layout (à la Llama): the head is split in two halves and
rotated together, which is numerically identical to rotating adjacent pairs.

    x, cos, sin shapes: x [B, n_head, T, head_size], cos/sin [T, head_size]
"""

from __future__ import annotations

import torch


def build_rope_cache(
    seq_len: int, head_size: int, base: float = 10000.0, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables of shape ``[seq_len, head_size]``."""
    if head_size % 2 != 0:
        raise ValueError(f"head_size must be even for RoPE, got {head_size}")
    # Frequencies for each pair: θ_i = base^(-2i/head_size)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_size, 2, dtype=torch.float32, device=device) / head_size)
    )  # [head_size/2]
    t = torch.arange(seq_len, dtype=torch.float32, device=device)  # [T]
    freqs = torch.outer(t, inv_freq)  # [T, head_size/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [T, head_size] (duplicated for the two halves)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Map [x1, x2] -> [-x2, x1] over the last dimension (the two halves)."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` [B, n_head, T, head_size] by the position angles in cos/sin [T, head_size]."""
    cos = cos.to(x.dtype)[None, None, :, :]
    sin = sin.to(x.dtype)[None, None, :, :]
    return x * cos + rotate_half(x) * sin
