#!/usr/bin/env python3
"""Print a lightweight environment report for the Beeja language-model project.

The report logic lives in ``beeja.utils.env`` so it can be unit-tested. This
script adds ``src/`` to ``sys.path`` so it also runs before ``pip install -e .``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running straight from a fresh checkout (before an editable install).
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from beeja.utils.env import environment_report  # noqa: E402


def main() -> int:
    print(json.dumps(environment_report(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
