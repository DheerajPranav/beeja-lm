"""Character tokenizer: round trips, vocabulary, and error behaviour."""

from __future__ import annotations

import pytest

from beeja.data.char import CharTokenizer


def test_from_text_builds_sorted_unique_vocab():
    tok = CharTokenizer.from_text("banana")
    assert tok.itos == ["a", "b", "n"]
    assert tok.vocab_size == 3


def test_ascii_round_trip():
    text = "the seed grows."
    tok = CharTokenizer.from_text(text)
    assert tok.decode(tok.encode(text)) == text


def test_unicode_round_trip():
    text = "café — bīja 種 🌱"
    tok = CharTokenizer.from_text(text)
    assert tok.decode(tok.encode(text)) == text


def test_empty_string():
    tok = CharTokenizer.from_text("abc")
    assert tok.encode("") == []
    assert tok.decode([]) == ""


def test_unknown_character_raises():
    tok = CharTokenizer.from_text("abc")
    with pytest.raises(ValueError):
        tok.encode("z")


def test_duplicate_vocab_rejected():
    with pytest.raises(ValueError):
        CharTokenizer(["a", "a"])
