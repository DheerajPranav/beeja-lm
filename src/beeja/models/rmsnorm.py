"""RMSNorm — Root-Mean-Square layer normalization.

LayerNorm centres (subtract mean) *and* scales (divide by std). RMSNorm drops the
centring and just rescales by the root-mean-square of the activations:

    RMSNorm(x) = x / sqrt(mean(x²) + eps) · weight

It has no bias and no mean subtraction, so it is cheaper and — empirically —
works as well or better in Transformers. The statistic is computed in float32 for
stability, then cast back (matters under mixed precision).

    x shape: [..., dim]  ->  same shape
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        normed = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed.to(dtype) * self.weight
