"""Learning-rate schedule: linear warmup then cosine decay.

    step < warmup            : linear ramp 0 -> max_lr
    warmup <= step < max      : cosine decay max_lr -> min_lr
    step >= max_steps         : min_lr (floor)

Warmup avoids a large early step wrecking a freshly initialised model; cosine
decay anneals the rate smoothly so late training takes smaller, refining steps.
"""

from __future__ import annotations

import math


def lr_at(step: int, *, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    # Cosine over the decay window [warmup_steps, max_steps].
    denom = max(1, max_steps - warmup_steps)
    ratio = (step - warmup_steps) / denom  # 0 -> 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))  # 1 -> 0
    return min_lr + coeff * (max_lr - min_lr)
