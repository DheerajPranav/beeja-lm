"""Models: bigram baseline and the decoder-only Transformer (BeejaGPT)."""

from __future__ import annotations

from beeja.models.attention import MultiHeadSelfAttention, SingleHeadAttention
from beeja.models.bigram import BigramLanguageModel
from beeja.models.block import TransformerBlock
from beeja.models.config import (
    ModelConfig,
    beeja_3m_config,
    beeja_3m_modern_config,
    modern_smoke_config,
    smoke_config,
)
from beeja.models.gpt import BeejaGPT
from beeja.models.mlp import MLP
from beeja.models.rmsnorm import RMSNorm
from beeja.models.rope import apply_rope, build_rope_cache
from beeja.models.swiglu import SwiGLU

__all__ = [
    "BigramLanguageModel",
    "ModelConfig",
    "smoke_config",
    "beeja_3m_config",
    "modern_smoke_config",
    "beeja_3m_modern_config",
    "SingleHeadAttention",
    "MultiHeadSelfAttention",
    "MLP",
    "SwiGLU",
    "RMSNorm",
    "build_rope_cache",
    "apply_rope",
    "TransformerBlock",
    "BeejaGPT",
]
