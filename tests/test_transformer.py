"""BeejaGPT: forward shapes, causal leakage, generation, and tiny-batch overfit."""

from __future__ import annotations

import math

import torch

from beeja.models.config import smoke_config
from beeja.models.gpt import BeejaGPT
from beeja.training.basic import fit_batch
from beeja.utils import set_seed

VOCAB = 40


def _model() -> BeejaGPT:
    set_seed(0)
    return BeejaGPT(smoke_config(VOCAB))


def test_forward_shapes_and_finite_loss():
    model = _model()
    idx = torch.randint(VOCAB, (4, 16))
    targets = torch.randint(VOCAB, (4, 16))
    logits, loss = model(idx, targets)
    assert logits.shape == (4, 16, VOCAB)
    assert loss.ndim == 0 and math.isfinite(loss.item())
    # Random-init loss should sit near ln(vocab).
    assert abs(loss.item() - math.log(VOCAB)) < 1.0


def test_no_future_token_leakage_end_to_end():
    model = _model()
    model.eval()
    idx = torch.randint(VOCAB, (1, 8))
    logits_a, _ = model(idx)
    # Perturb a token at position 5; positions 0..4 must be unaffected.
    idx2 = idx.clone()
    idx2[0, 5] = (idx2[0, 5] + 1) % VOCAB
    logits_b, _ = model(idx2)
    assert torch.allclose(logits_a[:, :5, :], logits_b[:, :5, :], atol=1e-5)
    assert not torch.allclose(logits_a[:, 5, :], logits_b[:, 5, :])


def test_generation_shape_and_context_cropping():
    model = _model()
    start = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(start, 40, generator=torch.Generator().manual_seed(1))
    # 1 seed token + 40 new; block_size cropping means no forward exceeds context.
    assert out.shape == (1, 41)


def test_generation_is_reproducible_with_fixed_seed():
    model = _model()
    start = torch.zeros((1, 1), dtype=torch.long)
    a = model.generate(start, 20, generator=torch.Generator().manual_seed(7))
    b = model.generate(start, 20, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)


def test_tiny_batch_overfit():
    model = _model()
    gen = torch.Generator().manual_seed(0)
    x = torch.randint(VOCAB, (4, 16), generator=gen)
    y = torch.randint(VOCAB, (4, 16), generator=gen)
    losses = fit_batch(model, x, y, steps=300, lr=3e-3)
    assert losses[-1] < 0.05, f"expected near-zero overfit loss, got {losses[-1]:.4f}"
    assert losses[-1] < losses[0]
