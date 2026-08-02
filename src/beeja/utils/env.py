"""Environment report.

Reports the host architecture, Python, memory, and PyTorch accelerator
availability so we can reason about what will and will not fit on the local
8 GB Apple Silicon machine versus a remote GPU host. ``torch`` is imported
lazily so the report still works before dependencies are installed.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from typing import Any


def _run(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _memory_bytes() -> int | None:
    """Total physical memory in bytes, or None if it cannot be determined."""
    if platform.system() == "Darwin":
        value = _run(["sysctl", "-n", "hw.memsize"])
        return int(value) if value and value.isdigit() else None

    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        return int(page_size * pages)
    except (AttributeError, OSError, ValueError):
        return None


def environment_report() -> dict[str, Any]:
    """Collect a lightweight, JSON-serialisable environment summary."""
    memory_bytes = _memory_bytes()
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "memory_bytes": memory_bytes,
        "memory_gib": round(memory_bytes / 1024**3, 2) if memory_bytes else None,
        "colab": "COLAB_RELEASE_TAG" in os.environ,
    }

    try:
        import torch

        mps_backend = getattr(torch.backends, "mps", None)
        report["torch"] = torch.__version__
        report["cuda_available"] = torch.cuda.is_available()
        report["mps_built"] = bool(mps_backend is not None and mps_backend.is_built())
        report["mps_available"] = bool(mps_backend is not None and mps_backend.is_available())
    except ImportError:
        report["torch"] = None
        report["cuda_available"] = False
        report["mps_built"] = False
        report["mps_available"] = False

    return report
