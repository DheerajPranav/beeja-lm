#!/usr/bin/env python3
"""Reproducibly download a small public-domain corpus for pretraining.

Fetches the Tiny Shakespeare text (a single UTF-8 file, ~1.1 MB) into ``data/``.
This is NOT run automatically — invoke it explicitly before a full training run.

    python scripts/download_data.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _ROOT / "data"
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DEST = DATA_DIR / "tinyshakespeare.txt"


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        print(f"already present: {DEST.relative_to(_ROOT)} ({DEST.stat().st_size} bytes)")
        return 0
    print(f"downloading {URL}")
    try:
        urllib.request.urlretrieve(URL, DEST)  # noqa: S310 (trusted, pinned URL)
    except OSError as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        return 1
    print(f"saved -> {DEST.relative_to(_ROOT)} ({DEST.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
