"""Beeja: an educational decoder-only language-model family.

The first named checkpoint is ``Beeja-3M``. Everything in this package is built
from first principles (raw PyTorch tensor ops and ``torch.nn`` building blocks)
so each mechanism stays visible and testable.

See ``CLAUDE.md`` for the project rules and ``PROJECT_STATE.md`` for the
current milestone.
"""

from __future__ import annotations

from beeja.utils import environment_report, select_device, set_seed

__all__ = [
    "__version__",
    "MODEL_FAMILY",
    "FIRST_CHECKPOINT",
    "select_device",
    "set_seed",
    "environment_report",
]

__version__ = "0.0.1"
MODEL_FAMILY = "Beeja"
FIRST_CHECKPOINT = "Beeja-3M"
