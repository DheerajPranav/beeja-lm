#!/usr/bin/env python3
"""Reproducibly download a public corpus for pretraining.

Datasets:
  tinystories    GPT-generated simple stories (Eldan & Li, 2023), built for tiny
                 models. Downloads the validation split (~20 MB, plenty for a
                 first char-level run); add --full for the ~2 GB train split.
  tinyshakespeare  The classic ~1 MB Shakespeare text.

This is NOT run automatically — invoke it explicitly before a training run.

    python scripts/download_data.py --dataset tinystories
    python scripts/download_data.py --dataset tinystories --full
    python scripts/download_data.py --dataset tinyshakespeare
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _ROOT / "data"

_TS = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"
DATASETS: dict[str, dict[str, str]] = {
    "tinyshakespeare": {
        "tinyshakespeare.txt": (
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
            "tinyshakespeare/input.txt"
        ),
    },
    "tinystories": {
        "TinyStories-valid.txt": f"{_TS}/TinyStories-valid.txt",
    },
}
# Large optional splits, only fetched with --full.
FULL_EXTRAS: dict[str, dict[str, str]] = {
    "tinystories": {"TinyStories-train.txt": f"{_TS}/TinyStories-train.txt"},
}


def _download(name: str, url: str) -> int:
    dest = DATA_DIR / name
    if dest.exists():
        print(f"already present: {dest.relative_to(_ROOT)} ({dest.stat().st_size:,} bytes)")
        return 0
    print(f"downloading {name} <- {url}")
    try:
        urllib.request.urlretrieve(url, dest)  # noqa: S310 (trusted, pinned URLs)
    except OSError as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        return 1
    print(f"saved -> {dest.relative_to(_ROOT)} ({dest.stat().st_size:,} bytes)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="tinystories")
    parser.add_argument("--full", action="store_true", help="also fetch large train split")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    files = dict(DATASETS[args.dataset])
    if args.full:
        files.update(FULL_EXTRAS.get(args.dataset, {}))

    rc = 0
    for name, url in files.items():
        rc |= _download(name, url)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
