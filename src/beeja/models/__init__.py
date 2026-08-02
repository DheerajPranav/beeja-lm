"""Models: bigram baseline and the decoder-only Transformer (BeejaGPT)."""

from __future__ import annotations

from beeja.models.attention import MultiHeadSelfAttention, SingleHeadAttention
from beeja.models.bigram import BigramLanguageModel
from beeja.models.block import TransformerBlock
from beeja.models.config import ModelConfig, beeja_3m_config, smoke_config
from beeja.models.gpt import BeejaGPT
from beeja.models.mlp import MLP

__all__ = [
    "BigramLanguageModel",
    "ModelConfig",
    "smoke_config",
    "beeja_3m_config",
    "SingleHeadAttention",
    "MultiHeadSelfAttention",
    "MLP",
    "TransformerBlock",
    "BeejaGPT",
]
