"""Quality metrics: perplexity / bits-per-token, and repetition / diversity.

Perplexity is exp(mean cross-entropy) — the effective number of equally-likely
choices the model is deciding between per token. Bits-per-token is the same loss
in base 2. Both are computed under no-grad and never update the model.

Repetition/diversity look at *generated* text: distinct-n is the fraction of
unique n-grams (higher = more varied); repetition-n is its complement.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch

from beeja.data.dataset import get_batch


@torch.no_grad()
def evaluate_perplexity(
    model: torch.nn.Module,
    data: torch.Tensor,
    *,
    block_size: int,
    batch_size: int,
    batches: int = 50,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """Mean loss, perplexity, and bits-per-token over ``batches`` random windows."""
    was_training = model.training
    model.eval()
    model.to(device)
    total = 0.0
    for _ in range(batches):
        x, y = get_batch(
            data, block_size=block_size, batch_size=batch_size, generator=generator, device=device
        )
        _, loss = model(x, y)
        total += loss.item()
    model.train(was_training)
    mean = total / batches
    return {
        "loss": mean,
        "perplexity": math.exp(mean),
        "bits_per_token": mean / math.log(2),
    }


def ngram_distinct(ids: Sequence[int], n: int) -> float:
    """Fraction of distinct n-grams in ``ids`` (0..1). 1.0 = all n-grams unique."""
    if len(ids) < n:
        return 0.0
    grams = [tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)]
    return len(set(grams)) / len(grams)


def diversity_report(ids: Sequence[int]) -> dict[str, Any]:
    """Distinct-1/2/3 and repetition-1 for a generated id sequence."""
    return {
        "distinct_1": round(ngram_distinct(ids, 1), 4),
        "distinct_2": round(ngram_distinct(ids, 2), 4),
        "distinct_3": round(ngram_distinct(ids, 3), 4),
        "repetition_1": round(1.0 - ngram_distinct(ids, 1), 4),
        "length": len(ids),
    }
