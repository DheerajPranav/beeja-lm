"""Dataset encoding, train/val split, and batch sampling."""

from __future__ import annotations

import pytest
import torch

from beeja.data.char import CharTokenizer
from beeja.data.dataset import encode_dataset, get_batch, train_val_split
from beeja.data.sample import SAMPLE_TEXT

TOK = CharTokenizer.from_text(SAMPLE_TEXT)
DATA = encode_dataset(SAMPLE_TEXT, TOK)


def test_encode_dataset_is_long_1d():
    assert DATA.dtype == torch.long
    assert DATA.ndim == 1
    assert len(DATA) == len(SAMPLE_TEXT)


def test_split_is_contiguous_and_non_overlapping():
    train, val = train_val_split(DATA, val_fraction=0.1)
    assert len(train) + len(val) == len(DATA)
    # val is exactly the tail of the stream (no shuffling, no leakage)
    assert torch.equal(torch.cat([train, val]), DATA)


def test_get_batch_shapes_and_target_shift():
    gen = torch.Generator().manual_seed(0)
    x, y = get_batch(DATA, block_size=8, batch_size=4, generator=gen)
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    # y must be x advanced by one position within the source stream: for each
    # sampled window, y[:, :-1] equals x[:, 1:].
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_get_batch_is_reproducible_with_fixed_generator():
    x1, y1 = get_batch(DATA, block_size=8, batch_size=4, generator=torch.Generator().manual_seed(3))
    x2, y2 = get_batch(DATA, block_size=8, batch_size=4, generator=torch.Generator().manual_seed(3))
    assert torch.equal(x1, x2) and torch.equal(y1, y2)


def test_get_batch_rejects_oversized_block():
    with pytest.raises(ValueError):
        get_batch(DATA, block_size=len(DATA), batch_size=2)


def test_split_rejects_bad_fraction():
    with pytest.raises(ValueError):
        train_val_split(DATA, val_fraction=1.5)
