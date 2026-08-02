"""Model configuration for the Beeja decoder-only Transformer.

A single dataclass describes the whole architecture so models are reproducible
and checkpoints can carry their own shape. Two factory configs are provided:

- ``smoke_config`` — tiny, for unit tests and fast overfit checks.
- ``beeja_3m_config`` — the first named release, tuned to land near 3M params.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int = 128  # maximum context length (positions)
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = True  # bias in Linear / LayerNorm layers

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
        for field in ("vocab_size", "block_size", "n_layer", "n_head", "n_embd"):
            if getattr(self, field) <= 0:
                raise ValueError(f"{field} must be positive, got {getattr(self, field)}")

    @property
    def head_size(self) -> int:
        return self.n_embd // self.n_head


def smoke_config(vocab_size: int) -> ModelConfig:
    """Tiny config for tests and smoke runs (~0.1M params)."""
    return ModelConfig(
        vocab_size=vocab_size, block_size=32, n_layer=2, n_head=4, n_embd=64, dropout=0.0
    )


def beeja_3m_config(vocab_size: int) -> ModelConfig:
    """The Beeja-3M release config.

    Reference ``architecture.md`` lists 4 layers / d=128 as a *starting point*,
    but that measures ~0.8M params. To make the ``Beeja-3M`` name honest we tune
    the model dimension to 256 (heads stay at 4, head_size 64), which measures
    ~3.2M — close to the 3M target. See LEARNING_LOG for the calculation.
    """
    return ModelConfig(
        vocab_size=vocab_size, block_size=128, n_layer=4, n_head=4, n_embd=256, dropout=0.0
    )
