"""Load a trained checkpoint for inference and stream completions.

Shared by the generate/evaluate CLIs and the interactive app. Rebuilds the
tokenizer from the training config and the model from the config stored inside
the checkpoint, with clear errors for the common failure cases.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
import yaml

from beeja.data.pipeline import build_datasets
from beeja.models.config import ModelConfig
from beeja.models.gpt import BeejaGPT

# Hard ceiling so a bad --max-new-tokens can't run away.
MAX_NEW_TOKENS_LIMIT = 4096


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def build_tokenizer(cfg: dict[str, Any]):
    """Rebuild the tokenizer exactly as training did (from the config's data section)."""
    d = cfg.get("data", {})
    _, _, tokenizer = build_datasets(
        source=d.get("source", "sample"),
        tokenizer_kind=d.get("tokenizer", "char"),
        vocab_size=d.get("vocab_size", 320),
        val_fraction=d.get("val_fraction", 0.1),
        max_chars=d.get("max_chars"),
        tokenizer_max_chars=d.get("tokenizer_max_chars"),
    )
    return tokenizer


def load_model(checkpoint_path: str | Path, device: str = "cpu") -> tuple[BeejaGPT, dict[str, Any]]:
    """Rebuild the model from a checkpoint. Raises clearly on the usual mistakes."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # Trainer checkpoints store "model_config"; simple checkpoints store "config".
    model_cfg = ckpt.get("model_config") or ckpt.get("config") if isinstance(ckpt, dict) else None
    if model_cfg is None or "model_state" not in ckpt:
        raise ValueError(f"not a Beeja checkpoint (missing model config/state): {path}")
    model = BeejaGPT(ModelConfig(**model_cfg))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt


def stream_completion(
    model: BeejaGPT,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int | None = None,
    device: str = "cpu",
    generator: torch.Generator | None = None,
) -> Iterator[str]:
    """Yield decoded text deltas as the model generates (KV-cached)."""
    n = max(1, min(int(max_new_tokens), MAX_NEW_TOKENS_LIMIT))  # safe bound
    ids = tokenizer.encode(prompt) if prompt else [0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    produced: list[int] = []
    prev = ""
    for tok in model.generate_stream(
        idx, n, temperature=temperature, top_k=top_k, generator=generator
    ):
        produced.append(int(tok.item()))
        text = tokenizer.decode(produced)  # decode all-so-far; emit only the new suffix
        yield text[len(prev) :]
        prev = text
