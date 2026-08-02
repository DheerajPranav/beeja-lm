"""Compression measurement for a trained tokenizer.

Compression ratio here means *bytes per token*: how many raw UTF-8 bytes the
average token stands in for. A byte-level model with no merges scores ~1.0
(one token per byte); learning merges raises it as common byte sequences fold
into single tokens. Measure it on held-out text, never the training text.
"""

from __future__ import annotations

from typing import Any

from beeja.tokenizer.bpe import BPETokenizer


def compression_report(tokenizer: BPETokenizer, text: str) -> dict[str, Any]:
    num_bytes = len(text.encode("utf-8"))
    num_chars = len(text)
    num_tokens = len(tokenizer.encode(text))
    return {
        "chars": num_chars,
        "bytes": num_bytes,
        "tokens": num_tokens,
        "bytes_per_token": round(num_bytes / num_tokens, 4) if num_tokens else 0.0,
        "chars_per_token": round(num_chars / num_tokens, 4) if num_tokens else 0.0,
    }
