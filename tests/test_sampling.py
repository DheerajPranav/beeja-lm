"""Next-token sampling: validation, top-k restriction, and reproducibility."""

from __future__ import annotations

import pytest
import torch

from beeja.generation.sampling import sample_next_token


def test_output_shape():
    logits = torch.randn(5, 10)
    out = sample_next_token(logits)
    assert out.shape == (5, 1)
    assert out.dtype == torch.long


def test_temperature_must_be_positive():
    with pytest.raises(ValueError):
        sample_next_token(torch.randn(2, 4), temperature=0.0)


def test_top_k_must_be_positive():
    with pytest.raises(ValueError):
        sample_next_token(torch.randn(2, 4), top_k=0)


def test_top_k_one_is_argmax():
    logits = torch.tensor([[0.1, 5.0, 0.2, -1.0]])
    out = sample_next_token(logits, top_k=1)
    assert out.item() == 1  # the clear maximum


def test_reproducible_with_fixed_generator():
    logits = torch.randn(3, 8)
    a = sample_next_token(logits, generator=torch.Generator().manual_seed(42))
    b = sample_next_token(logits, generator=torch.Generator().manual_seed(42))
    assert torch.equal(a, b)


def test_rejects_non_2d_logits():
    with pytest.raises(ValueError):
        sample_next_token(torch.randn(4))
