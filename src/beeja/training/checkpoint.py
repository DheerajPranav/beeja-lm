"""Resumable checkpointing for BeejaGPT.

A checkpoint is a single ``torch.save`` dict that carries the model's own config
(so it can be rebuilt without external metadata), the weights, the release name,
and an optional training-state payload (step, optimizer, RNG) for resume.

    save_checkpoint(path, model, name="Beeja-3M", step=..., extra={...})
    model, ckpt = load_checkpoint(path)
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from beeja.models.config import ModelConfig
from beeja.models.gpt import BeejaGPT

FORMAT_VERSION = 1


def save_checkpoint(
    path: str | Path,
    model: BeejaGPT,
    *,
    name: str,
    step: int = 0,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "name": name,
        "step": step,
        "config": asdict(model.config),
        "model_state": model.state_dict(),
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> tuple[BeejaGPT, dict[str, Any]]:
    """Rebuild the model from its stored config and load weights. Returns (model, ckpt)."""
    # weights_only=False: this is our own trusted checkpoint containing a config dict.
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    config = ModelConfig(**ckpt["config"])
    model = BeejaGPT(config)
    model.load_state_dict(ckpt["model_state"])
    model.to(map_location)
    model.eval()
    return model, ckpt
