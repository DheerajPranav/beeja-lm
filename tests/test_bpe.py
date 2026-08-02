"""Byte-level BPE: round trips, determinism, save/load, specials, compression."""

from __future__ import annotations

import pytest

from beeja.data.sample import SAMPLE_TEXT
from beeja.tokenizer.bpe import BPETokenizer, get_stats, merge
from beeja.tokenizer.report import compression_report


def _trained(vocab_size: int = 320) -> BPETokenizer:
    tok = BPETokenizer()
    tok.train(SAMPLE_TEXT, vocab_size=vocab_size)
    return tok


# -- helpers ----------------------------------------------------------------
def test_get_stats_and_merge():
    ids = [1, 2, 1, 2, 3]
    stats = get_stats(ids)
    assert stats[(1, 2)] == 2
    assert merge(ids, (1, 2), 99) == [99, 99, 3]


# -- round trips ------------------------------------------------------------
def test_ascii_round_trip():
    tok = _trained()
    text = "A seed is a small promise."
    assert tok.decode(tok.encode(text)) == text


def test_unicode_round_trip():
    tok = _trained()
    text = "café — bīja 種子 🌱🌳 naïve"
    # Byte-level means no unknown tokens even for text unseen during training.
    assert tok.decode(tok.encode(text)) == text


def test_untrained_tokenizer_is_identity_over_bytes():
    tok = BPETokenizer()  # no merges: one token per UTF-8 byte
    text = "hi 🌱"
    ids = tok.encode(text)
    assert ids == list(text.encode("utf-8"))
    assert tok.decode(ids) == text


def test_empty_string():
    tok = _trained()
    assert tok.encode("") == []
    assert tok.decode([]) == ""


# -- determinism & sizing ---------------------------------------------------
def test_training_is_deterministic():
    a = _trained()
    b = _trained()
    assert a.merges == b.merges
    assert a.vocab == b.vocab


def test_vocab_size_requires_byte_base():
    with pytest.raises(ValueError):
        BPETokenizer().train("hello", vocab_size=100)


def test_merges_actually_shorten_the_sequence():
    tok = _trained()
    ids = tok.encode(SAMPLE_TEXT)
    assert len(ids) < len(SAMPLE_TEXT.encode("utf-8"))


# -- save / load ------------------------------------------------------------
def test_save_load_identity(tmp_path):
    tok = _trained()
    tok.register_special_tokens(["<|endoftext|>"])
    path = tmp_path / "bpe.json"
    tok.save(path)

    reloaded = BPETokenizer.from_file(path)
    assert reloaded.merges == tok.merges
    assert reloaded.vocab == tok.vocab
    assert reloaded.special_tokens == tok.special_tokens
    text = "the seed becomes a forest 🌳"
    assert reloaded.encode(text) == tok.encode(text)


# -- special tokens ---------------------------------------------------------
def test_special_token_recognised_only_when_allowed():
    tok = _trained()
    tok.register_special_tokens(["<|endoftext|>"])
    eot = tok.special_tokens["<|endoftext|>"]

    # default: treated as ordinary text (no special id emitted)
    assert eot not in tok.encode("a<|endoftext|>b")
    # allowed: emitted as a single special id, and decode reconstructs it
    ids = tok.encode("a<|endoftext|>b", allowed_special="all")
    assert eot in ids
    assert tok.decode(ids) == "a<|endoftext|>b"


def test_decode_unknown_id_raises():
    tok = _trained()
    with pytest.raises(ValueError):
        tok.decode([10**9])


# -- compression ------------------------------------------------------------
def test_compression_report_beats_one_byte_per_token():
    tok = _trained()
    report = compression_report(tok, SAMPLE_TEXT)
    assert report["bytes"] == len(SAMPLE_TEXT.encode("utf-8"))
    assert report["bytes_per_token"] > 1.0  # merges pack multiple bytes per token
