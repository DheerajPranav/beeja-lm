#!/usr/bin/env python3
"""Train the byte-level BPE tokenizer on the sample corpus and report compression.

Trains on the train split, saves the tokenizer, and measures compression on the
held-out val split (never the training text). The corpus is intentionally tiny
at this stage; this demonstrates the mechanism, not production-scale compression.

    python scripts/train_tokenizer.py --vocab-size 320
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from beeja.data.sample import SAMPLE_TEXT  # noqa: E402
from beeja.tokenizer.bpe import BPETokenizer  # noqa: E402
from beeja.tokenizer.report import compression_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=int, default=320)
    args = parser.parse_args()

    split = int(len(SAMPLE_TEXT) * 0.9)
    train_text, val_text = SAMPLE_TEXT[:split], SAMPLE_TEXT[split:]

    tok = BPETokenizer()
    tok.train(train_text, vocab_size=args.vocab_size)
    tok.register_special_tokens(["<|endoftext|>"])
    print(f"learned {len(tok.merges)} merges; vocab_size={tok.vocab_size}")

    out = _ROOT / "checkpoints" / "beeja-bpe.json"
    tok.save(out)
    print(f"saved tokenizer -> {out.relative_to(_ROOT)}")

    print("compression (held-out val text):")
    print(json.dumps(compression_report(tok, val_text), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
