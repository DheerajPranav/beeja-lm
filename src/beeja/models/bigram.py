"""Bigram language model — the baseline before any attention.

The model is a single lookup table of shape ``[vocab, vocab]``: row ``i`` holds
the logits for the token that follows token ``i``. Prediction depends only on
the *immediately preceding* token (a bigram), so it cannot use longer context —
which is exactly why it is the right thing to beat once we add self-attention.

Objective (next-token cross-entropy), for a sequence ``x_1..x_T``:

    L = -(1/T) * sum_t log p(x_t | x_{t-1})

Shapes:

    idx     : [B, T]        token ids
    logits  : [B, T, V]     one distribution per position
    targets : [B, T]        next-token ids
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from beeja.generation.sampling import sample_next_token


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        self.vocab_size = vocab_size
        # Embedding(V, V): looking up token i returns its next-token logits.
        self.token_logits = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = self.token_logits(idx)  # [B, T, V]
        loss: torch.Tensor | None = None
        if targets is not None:
            b, t, v = logits.shape
            # Flatten batch and time so cross_entropy sees [B*T, V] vs [B*T].
            loss = F.cross_entropy(logits.view(b * t, v), targets.view(b * t))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Autoregressively extend ``idx`` (shape ``[B, T]``) by ``max_new_tokens``."""
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(idx)  # [B, T, V]
            next_logits = logits[:, -1, :]  # only the last step matters — [B, V]
            next_id = sample_next_token(
                next_logits, temperature=temperature, top_k=top_k, generator=generator
            )
            idx = torch.cat([idx, next_id], dim=1)
        return idx
