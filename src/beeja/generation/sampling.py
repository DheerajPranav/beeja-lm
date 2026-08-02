"""Next-token sampling controls: temperature and top-k.

Given the model's logits for the next token, we turn them into a probability
distribution and draw one sample. Two knobs shape that distribution:

- **temperature** ``T`` divides the logits before softmax. ``T -> 0`` sharpens
  toward greedy (argmax); ``T > 1`` flattens toward uniform.
- **top-k** keeps only the ``k`` highest-scoring tokens and renormalises, so the
  tail of unlikely tokens can never be sampled.

    logits: [batch, vocab]  -->  next ids: [batch, 1]
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one token id per row of ``logits``.

    Args:
        logits: Unnormalised scores, shape ``[batch, vocab]``.
        temperature: Positive scalar; smaller is greedier.
        top_k: If set, restrict sampling to the ``k`` most likely tokens.
        generator: Optional RNG (same device as ``logits``) for reproducibility.

    Returns:
        Long tensor of shape ``[batch, 1]``.
    """
    if logits.ndim != 2:
        raise ValueError(f"expected logits of shape [batch, vocab], got {tuple(logits.shape)}")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    logits = logits / temperature

    if top_k is not None:
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        k = min(top_k, logits.size(-1))
        kth_value = torch.topk(logits, k, dim=-1).values[:, -1, None]  # [batch, 1]
        logits = logits.masked_fill(logits < kth_value, float("-inf"))

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)
