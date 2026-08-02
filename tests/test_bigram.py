"""Bigram model: shapes, a real training step, loss decrease, and generation."""

from __future__ import annotations

import math

import torch

from beeja.data.char import CharTokenizer
from beeja.data.dataset import encode_dataset, train_val_split
from beeja.data.sample import SAMPLE_TEXT
from beeja.models.bigram import BigramLanguageModel
from beeja.training.basic import fit
from beeja.utils import set_seed

TOK = CharTokenizer.from_text(SAMPLE_TEXT)
DATA = encode_dataset(SAMPLE_TEXT, TOK)


def test_forward_shapes_and_finite_loss():
    model = BigramLanguageModel(TOK.vocab_size)
    x = DATA[:32].view(4, 8)
    y = DATA[1:33].view(4, 8)
    logits, loss = model(x, y)
    assert logits.shape == (4, 8, TOK.vocab_size)
    assert loss.ndim == 0 and math.isfinite(loss.item())


def test_untrained_loss_near_uniform():
    # With random init, loss should be close to ln(vocab_size) (uniform guess).
    set_seed(0)
    model = BigramLanguageModel(TOK.vocab_size)
    x = DATA[:64].view(8, 8)
    y = DATA[1:65].view(8, 8)
    _, loss = model(x, y)
    assert loss.item() < math.log(TOK.vocab_size) + 1.0


def test_one_step_changes_parameters():
    set_seed(0)
    model = BigramLanguageModel(TOK.vocab_size)
    before = model.token_logits.weight.detach().clone()
    fit(
        model,
        DATA,
        block_size=8,
        batch_size=8,
        steps=1,
        lr=1e-1,
        generator=torch.Generator().manual_seed(0),
    )
    assert not torch.equal(before, model.token_logits.weight.detach())


def test_loss_decreases_on_tiny_dataset():
    set_seed(0)
    train_data, _ = train_val_split(DATA, val_fraction=0.1)
    model = BigramLanguageModel(TOK.vocab_size)
    losses = fit(
        model,
        train_data,
        block_size=16,
        batch_size=32,
        steps=400,
        lr=1e-2,
        generator=torch.Generator().manual_seed(0),
    )
    first = sum(losses[:20]) / 20
    last = sum(losses[-20:]) / 20
    assert last < first, f"expected loss to drop, first={first:.3f} last={last:.3f}"


def test_generation_shape_and_fixed_seed_reproducible():
    set_seed(0)
    model = BigramLanguageModel(TOK.vocab_size)
    start = torch.zeros((1, 1), dtype=torch.long)
    a = model.generate(start, 20, generator=torch.Generator().manual_seed(7))
    b = model.generate(start, 20, generator=torch.Generator().manual_seed(7))
    assert a.shape == (1, 21)  # 1 seed token + 20 new
    assert torch.equal(a, b)
