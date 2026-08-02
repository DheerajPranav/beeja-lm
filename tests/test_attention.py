"""Attention: shapes, probabilities sum to one, and no future leakage."""

from __future__ import annotations

import pytest
import torch

from beeja.models.attention import MultiHeadSelfAttention, SingleHeadAttention
from beeja.models.config import ModelConfig
from beeja.utils import set_seed


def test_single_head_shape_and_probabilities():
    set_seed(0)
    head = SingleHeadAttention(n_embd=16, head_size=8, block_size=8)
    x = torch.randn(2, 8, 16)
    out, att = head(x, return_attn=True)
    assert out.shape == (2, 8, 8)
    assert att.shape == (2, 8, 8)
    # Each query row is a probability distribution over keys.
    assert torch.allclose(att.sum(dim=-1), torch.ones(2, 8), atol=1e-6)


def test_single_head_future_positions_have_zero_probability():
    set_seed(0)
    head = SingleHeadAttention(n_embd=16, head_size=8, block_size=8)
    _, att = head(torch.randn(2, 8, 16), return_attn=True)
    future = torch.triu(torch.ones(8, 8), diagonal=1).bool()  # strictly-upper = future keys
    assert torch.all(att[:, future] == 0)


def test_multi_head_shape_and_probabilities():
    set_seed(0)
    config = ModelConfig(vocab_size=32, block_size=8, n_layer=1, n_head=4, n_embd=16)
    attn = MultiHeadSelfAttention(config)
    x = torch.randn(3, 8, 16)
    out, att = attn(x, return_attn=True)
    assert out.shape == (3, 8, 16)  # same shape in, same shape out
    assert att.shape == (3, 4, 8, 8)  # [B, n_head, T, T]
    assert torch.allclose(att.sum(dim=-1), torch.ones(3, 4, 8), atol=1e-6)


def test_multi_head_no_future_leakage():
    set_seed(0)
    config = ModelConfig(vocab_size=32, block_size=8, n_layer=1, n_head=4, n_embd=16)
    attn = MultiHeadSelfAttention(config)
    _, att = attn(torch.randn(2, 8, 16), return_attn=True)
    future = torch.triu(torch.ones(8, 8), diagonal=1).bool()
    assert torch.all(att[:, :, future] == 0)


def test_invalid_head_divisibility_raises():
    with pytest.raises(ValueError):
        ModelConfig(vocab_size=32, n_head=3, n_embd=16)  # 16 % 3 != 0
