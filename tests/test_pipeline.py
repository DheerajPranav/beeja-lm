"""Data pipeline: subword streams, round trips, and no tokenizer leakage."""

from __future__ import annotations

import pytest
import torch

from beeja.data.pipeline import build_datasets, build_tokenizer, load_text


def test_char_pipeline_shapes_and_vocab():
    train_ids, val_ids, tok = build_datasets("sample", "char", val_fraction=0.1)
    assert train_ids.dtype == torch.long and val_ids.dtype == torch.long
    assert train_ids.ndim == 1 and val_ids.ndim == 1
    assert len(train_ids) > len(val_ids) > 0
    assert tok.vocab_size > 0


def test_bpe_pipeline_round_trips_each_split():
    text = load_text("sample")
    split = int(len(text) * 0.9)
    train_text, val_text = text[:split], text[split:]
    train_ids, val_ids, tok = build_datasets("sample", "bpe", vocab_size=320, val_fraction=0.1)
    # Each split decodes back to its own text (correct, separate encoding).
    assert tok.decode(train_ids.tolist()) == train_text
    assert tok.decode(val_ids.tolist()) == val_text


def test_tokenizer_is_trained_on_train_text_only():
    # A merge learned by the pipeline must be justified by the TRAIN text: the
    # pair's byte sequence should appear in train (no leakage from val).
    text = load_text("sample")
    split = int(len(text) * 0.9)
    train_text = text[:split]
    _, _, tok = build_datasets("sample", "bpe", vocab_size=300, val_fraction=0.1)
    train_bytes = train_text.encode("utf-8")
    for idx, byte_seq in tok.vocab.items():
        if idx >= 256:  # a learned merge
            assert byte_seq in train_bytes


def test_unknown_tokenizer_kind_raises():
    with pytest.raises(ValueError):
        build_tokenizer("wordpiece", "hello", vocab_size=300)
