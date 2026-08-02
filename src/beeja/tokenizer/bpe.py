"""A byte-level Byte-Pair Encoding (BPE) tokenizer, from scratch.

Why byte-level? Every string is first UTF-8 encoded to a sequence of bytes, so
the base vocabulary is exactly the 256 possible byte values. That means *any*
Unicode text — accents, CJK, emoji — is representable with no "unknown token",
and decode is always exact for valid input.

Training (learning merges):

1. Encode the corpus to a list of byte ids (0–255).
2. Count adjacent pairs; merge the most frequent pair into a new id (256, 257…).
3. Repeat until the target vocabulary size is reached.

Encoding applies the learned merges greedily, always merging the pair with the
lowest merge rank present. Decoding concatenates each id's byte string and
UTF-8 decodes.

This is the classic algorithm (à la Sennrich et al. 2016 / minbpe), implemented
transparently for learning — no tokenizer library is used.
"""

from __future__ import annotations

import json
from pathlib import Path

Pair = tuple[int, int]


def get_stats(ids: list[int], counts: dict[Pair, int] | None = None) -> dict[Pair, int]:
    """Count occurrences of each adjacent pair. Insertion order = first appearance."""
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:], strict=False):  # unequal lengths by design
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: list[int], pair: Pair, new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of ``pair`` with ``new_id``."""
    out: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self) -> None:
        self.merges: dict[Pair, int] = {}  # pair -> new id; insertion order = rank
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special_tokens: dict[str, int] = {}  # e.g. "<|endoftext|>" -> 320
        self.special_ids: dict[int, str] = {}

    # -- training -----------------------------------------------------------
    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Learn merges from ``text`` until the vocabulary reaches ``vocab_size``.

        Deterministic for identical input: the most-frequent pair is chosen by
        count, and ``max`` breaks ties by first appearance (stable dict order).
        """
        if vocab_size < 256:
            raise ValueError(f"vocab_size must be >= 256 (byte base), got {vocab_size}")
        num_merges = vocab_size - 256

        ids = list(text.encode("utf-8"))
        merges: dict[Pair, int] = {}
        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

        for i in range(num_merges):
            stats = get_stats(ids)
            if not stats:
                break  # nothing left to merge
            pair = max(stats, key=stats.get)  # type: ignore[arg-type]
            new_id = 256 + i
            ids = merge(ids, pair, new_id)
            merges[pair] = new_id
            vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(f"merge {i + 1}/{num_merges}: {pair} -> {new_id} ({vocab[new_id]!r})")

        self.merges = merges
        self.vocab = vocab

    # -- special tokens -----------------------------------------------------
    def register_special_tokens(self, tokens: list[str] | dict[str, int]) -> None:
        """Register special tokens with ids above the learned BPE vocabulary."""
        if isinstance(tokens, dict):
            self.special_tokens = dict(tokens)
        else:
            base = 256 + len(self.merges)
            self.special_tokens = {tok: base + i for i, tok in enumerate(tokens)}
        self.special_ids = {i: tok for tok, i in self.special_tokens.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.special_tokens)

    # -- encode / decode ----------------------------------------------------
    def _encode_chunk(self, ids: list[int]) -> list[int]:
        while len(ids) >= 2:
            stats = get_stats(ids)
            # Merge the pair with the lowest learned rank present; stop if none apply.
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def _encode_ordinary(self, text: str) -> list[int]:
        return self._encode_chunk(list(text.encode("utf-8")))

    def encode(self, text: str, allowed_special: str | set[str] = "none") -> list[int]:
        """Encode ``text`` to token ids.

        ``allowed_special``: ``"none"`` (default) treats special-token strings as
        ordinary text; ``"all"`` recognises every registered special token; a set
        recognises only those strings.
        """
        special = self._resolve_special(allowed_special)
        if not special:
            return self._encode_ordinary(text)

        import re

        pattern = "(" + "|".join(re.escape(s) for s in special) + ")"
        ids: list[int] = []
        for chunk in re.split(pattern, text):
            if chunk in special:
                ids.append(special[chunk])
            elif chunk:
                ids.extend(self._encode_ordinary(chunk))
        return ids

    def decode(self, ids: list[int]) -> str:
        parts: list[bytes] = []
        for i in ids:
            if i in self.vocab:
                parts.append(self.vocab[i])
            elif i in self.special_ids:
                parts.append(self.special_ids[i].encode("utf-8"))
            else:
                raise ValueError(f"token id {i} is not in the vocabulary")
        # errors="replace" guards against decoding an arbitrary/partial id stream;
        # for valid round-trips the bytes reconstruct the original exactly.
        return b"".join(parts).decode("utf-8", errors="replace")

    def _resolve_special(self, allowed_special: str | set[str]) -> dict[str, int]:
        if allowed_special == "none":
            return {}
        if allowed_special == "all":
            return self.special_tokens
        if isinstance(allowed_special, set):
            return {s: self.special_tokens[s] for s in allowed_special}
        raise ValueError(f"invalid allowed_special: {allowed_special!r}")

    # -- persistence --------------------------------------------------------
    def _build_vocab(self) -> dict[int, bytes]:
        vocab = {i: bytes([i]) for i in range(256)}
        for (p0, p1), idx in self.merges.items():  # merges ordered by rank
            vocab[idx] = vocab[p0] + vocab[p1]
        return vocab

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "merges": [[p0, p1, idx] for (p0, p1), idx in self.merges.items()],
            "special_tokens": self.special_tokens,
        }
        path.write_text(json.dumps(data))

    def load(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text())
        self.merges = {(p0, p1): idx for p0, p1, idx in data["merges"]}
        self.vocab = self._build_vocab()
        self.special_tokens = dict(data["special_tokens"])
        self.special_ids = {i: tok for tok, i in self.special_tokens.items()}

    @classmethod
    def from_file(cls, path: str | Path) -> BPETokenizer:
        tok = cls()
        tok.load(path)
        return tok
