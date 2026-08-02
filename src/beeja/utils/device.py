"""Device selection for Beeja.

The primary machine is an Apple Silicon M2 (8 GB unified memory), so the
preferred accelerator is Apple's Metal Performance Shaders (MPS) backend. On a
remote GPU host (for example Google Colab) CUDA is preferred. CPU is always the
final fallback so the code runs anywhere.

Priority order: MPS -> CUDA -> CPU.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

# Default preference order. MPS first for local Apple Silicon, CUDA for remote
# GPUs, CPU as the universal fallback.
DEFAULT_PRIORITY: tuple[str, ...] = ("mps", "cuda", "cpu")


def _is_available(kind: str) -> bool:
    """Return True if a backend of the given kind can be used right now."""
    if kind == "cpu":
        return True
    if kind == "cuda":
        return torch.cuda.is_available()
    if kind == "mps":
        backend = getattr(torch.backends, "mps", None)
        return bool(backend is not None and backend.is_available())
    raise ValueError(f"Unknown device kind: {kind!r}")


def select_device(prefer: Sequence[str] = DEFAULT_PRIORITY) -> torch.device:
    """Pick the best available :class:`torch.device` from ``prefer``.

    Args:
        prefer: Ordered backend names to try, e.g. ``("mps", "cuda", "cpu")``.

    Returns:
        The first available device in ``prefer``; CPU if none match.
    """
    for kind in prefer:
        if _is_available(kind):
            return torch.device(kind)
    return torch.device("cpu")
