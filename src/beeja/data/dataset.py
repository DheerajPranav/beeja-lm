"""Turn text into training batches for next-character prediction.

Pipeline:

    text --encode--> 1-D LongTensor of ids  (shape: [N])
    split into a contiguous train part and a held-out validation tail
    sample random windows to form batches

A batch is a pair ``(x, y)`` where ``y`` is ``x`` shifted one step to the left:
predicting ``y[t]`` from ``x[<=t]`` is exactly the next-token objective. Both
have shape ``[batch_size, block_size]``.
"""

from __future__ import annotations

import torch

from beeja.data.char import CharTokenizer


def encode_dataset(text: str, tokenizer: CharTokenizer) -> torch.Tensor:
    """Encode ``text`` into a 1-D ``LongTensor`` of token ids, shape ``[N]``."""
    return torch.tensor(tokenizer.encode(text), dtype=torch.long)


def train_val_split(
    data: torch.Tensor, val_fraction: float = 0.1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a 1-D id stream into contiguous ``(train, val)`` tensors.

    The split is positional (val is the tail), never shuffled, so no window can
    straddle the boundary and leak validation text into training.
    """
    if data.ndim != 1:
        raise ValueError(f"expected a 1-D tensor, got shape {tuple(data.shape)}")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    n_val = int(len(data) * val_fraction)
    split = len(data) - n_val
    return data[:split], data[split:]


def get_batch(
    data: torch.Tensor,
    *,
    block_size: int,
    batch_size: int,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of ``(x, y)`` windows.

    Returns:
        ``x`` and ``y`` each of shape ``[batch_size, block_size]``. ``y`` is
        ``x`` advanced by one position (the next-character targets).
    """
    if data.ndim != 1:
        raise ValueError(f"expected a 1-D tensor, got shape {tuple(data.shape)}")
    if len(data) <= block_size:
        raise ValueError(
            f"data length {len(data)} must exceed block_size {block_size} to form a window"
        )
    # Random start offsets on the CPU so a CPU generator gives reproducible batches
    # regardless of the target device.
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    if device is not None:
        x, y = x.to(device), y.to(device)
    return x, y
