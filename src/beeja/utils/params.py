"""Parameter counting and a memory caveat.

Reports total/trainable parameters and a breakdown by component (embeddings,
attention, MLP, LM head, norms/other), plus the raw parameter memory.

Important: parameter memory is a *lower bound* on training memory. Training also
holds gradients (~1x params), optimizer state (AdamW ~2x params), activations
(scale with batch × context × depth), and the attention matrices (∝ T²). Never
judge whether a model "fits" from parameter bytes alone.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn


def _bucket(name: str) -> str:
    if "token_emb" in name or "pos_emb" in name:
        return "embedding"
    if ".attn." in name:
        return "attention"
    if ".mlp." in name:
        return "mlp"
    if "lm_head" in name:
        return "lm_head"
    return "norm_other"


def parameter_count(model: nn.Module) -> dict[str, Any]:
    """Return a parameter-count breakdown and parameter memory for ``model``."""
    buckets = {"embedding": 0, "attention": 0, "mlp": 0, "lm_head": 0, "norm_other": 0}
    total = 0
    trainable = 0
    param_bytes = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
        param_bytes += n * p.element_size()
        buckets[_bucket(name)] += n

    return {
        "total": total,
        "trainable": trainable,
        **buckets,
        "param_bytes": param_bytes,
        "param_mib": round(param_bytes / 1024**2, 3),
    }
