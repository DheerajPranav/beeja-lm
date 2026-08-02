"""Deterministic seeding.

Reproducible small CPU tests are a quality gate for this project (see
``.claude/skills/beeja-lab/references/testing.md``). ``set_seed`` seeds every
random source we touch: Python ``hash`` randomization, the ``random`` module,
NumPy, and PyTorch (CPU, CUDA, and MPS).
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """Seed all relevant RNGs for reproducible runs.

    Args:
        seed: Non-negative integer seed.
        deterministic: If True, also request deterministic algorithms. This can
            be slower and may error on ops without a deterministic kernel, so it
            is opt-in (``warn_only=True`` downgrades those errors to warnings).

    Raises:
        ValueError: If ``seed`` is negative.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # seeds the CPU generator (and CUDA if present)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        torch.mps.manual_seed(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
