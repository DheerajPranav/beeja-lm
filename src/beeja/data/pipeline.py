"""End-to-end data pipeline: text -> tokenizer -> train/val id streams.

Leakage guard: the corpus is split *first*. What counts as leakage is *learned
statistics*, not the symbol space:

- BPE **merges** are learned, so they are trained on the **train** portion only;
  the val text is encoded with them but never shapes them.
- The **alphabet** (char set) and the 256-byte BPE base are the representable
  symbol space, not learned information — the char vocabulary is therefore built
  from the full corpus so val characters are always representable.

Works with either tokenizer — both ``CharTokenizer`` and ``BPETokenizer`` expose
``.encode`` / ``.vocab_size``.
"""

from __future__ import annotations

from pathlib import Path

import torch

from beeja.data.char import CharTokenizer
from beeja.data.sample import SAMPLE_TEXT
from beeja.tokenizer.bpe import BPETokenizer


def load_text(source: str) -> str:
    """``"sample"`` returns the embedded corpus; anything else is a file path."""
    if source == "sample":
        return SAMPLE_TEXT
    return Path(source).read_text(encoding="utf-8")


def build_tokenizer(kind: str, train_text: str, vocab_size: int = 320):
    if kind == "char":
        return CharTokenizer.from_text(train_text)
    if kind == "bpe":
        tok = BPETokenizer()
        tok.train(train_text, vocab_size=vocab_size)
        return tok
    raise ValueError(f"unknown tokenizer kind: {kind!r}")


def build_datasets(
    source: str = "sample",
    tokenizer_kind: str = "char",
    vocab_size: int = 320,
    val_fraction: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, object]:
    """Return ``(train_ids, val_ids, tokenizer)`` with no tokenizer leakage.

    Both id tensors are 1-D ``LongTensor``s ready for ``get_batch``.
    """
    text = load_text(source)
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    split = int(len(text) * (1.0 - val_fraction))
    train_text, val_text = text[:split], text[split:]

    # char: alphabet from the full corpus (symbol space, not leakage).
    # bpe:  merges learned from train text only (learned statistics).
    tok_source = text if tokenizer_kind == "char" else train_text
    tokenizer = build_tokenizer(tokenizer_kind, tok_source, vocab_size)
    train_ids = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)
    return train_ids, val_ids, tokenizer
