"""A minimal training loop used by the bigram baseline and smoke tests.

Deliberately small: sample a batch, compute cross-entropy, step AdamW. Larger
stages add warm-up, cosine decay, gradient clipping, accumulation, checkpointing,
and resume — none of which the bigram baseline needs yet.
"""

from __future__ import annotations

import torch

from beeja.data.dataset import get_batch


def fit(
    model: torch.nn.Module,
    train_data: torch.Tensor,
    *,
    block_size: int,
    batch_size: int,
    steps: int,
    lr: float = 1e-2,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> list[float]:
    """Train ``model`` for ``steps`` steps; return the per-step loss history."""
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    losses: list[float] = []
    for _ in range(steps):
        x, y = get_batch(
            train_data,
            block_size=block_size,
            batch_size=batch_size,
            generator=generator,
            device=device,
        )
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def fit_batch(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    steps: int,
    lr: float = 3e-3,
    device: torch.device | str = "cpu",
) -> list[float]:
    """Overfit a single fixed ``(x, y)`` batch; return the per-step loss history.

    Used by the tiny-batch overfit gate: a correct model with enough capacity
    should drive the loss on one memorised batch close to zero.
    """
    model.to(device)
    model.train()
    x, y = x.to(device), y.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses: list[float] = []
    for _ in range(steps):
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    data: torch.Tensor,
    *,
    block_size: int,
    batch_size: int,
    batches: int = 20,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> float:
    """Mean loss over ``batches`` random windows, without updating the model."""
    model.to(device)
    was_training = model.training
    model.eval()
    total = 0.0
    for _ in range(batches):
        x, y = get_batch(
            data,
            block_size=block_size,
            batch_size=batch_size,
            generator=generator,
            device=device,
        )
        _, loss = model(x, y)
        total += loss.item()
    model.train(was_training)
    return total / batches
