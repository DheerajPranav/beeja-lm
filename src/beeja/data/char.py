"""Character-level tokenizer.

The simplest possible tokenizer: the vocabulary is the sorted set of unique
characters in a corpus, and a token id is just that character's index. It has no
notion of subwords, but it is perfect for the bigram baseline because it makes
the model's job (predict the next character) completely transparent.

    encode: str        -> list[int]   (one id per character)
    decode: list[int]  -> str
"""

from __future__ import annotations

from collections.abc import Iterable


class CharTokenizer:
    """Maps characters to integer ids and back, built from a fixed vocabulary."""

    def __init__(self, chars: Iterable[str]) -> None:
        self.itos: list[str] = list(chars)
        if len(set(self.itos)) != len(self.itos):
            raise ValueError("vocabulary contains duplicate characters")
        self.stoi: dict[str, int] = {ch: i for i, ch in enumerate(self.itos)}

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        """Build a tokenizer from the sorted unique characters of ``text``."""
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        try:
            return [self.stoi[ch] for ch in text]
        except KeyError as exc:  # character never seen when the vocab was built
            raise ValueError(f"character not in vocabulary: {exc.args[0]!r}") from exc

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.itos[i] for i in ids)
