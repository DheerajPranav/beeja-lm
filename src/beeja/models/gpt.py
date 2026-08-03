"""BeejaGPT — the decoder-only autoregressive Transformer.

Forward pass (shapes):

    idx        [B, T]              input token ids
    tok + pos  [B, T, C]           token embedding + position embedding
    blocks     [B, T, C]           N pre-norm Transformer blocks
    ln_f       [B, T, C]           final LayerNorm
    lm_head    [B, T, V]           logits over the vocabulary

Objective: next-token cross-entropy,  L = -(1/T) Σ_t log p(x_t | x_<t).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from beeja.generation.sampling import sample_next_token
from beeja.models.block import TransformerBlock, make_norm
from beeja.models.config import ModelConfig


class BeejaGPT(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        # Learned absolute positions; None when RoPE handles position inside attention.
        self.pos_emb = (
            nn.Embedding(config.block_size, config.n_embd)
            if config.pos_encoding == "learned"
            else None
        )
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = make_norm(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        # GPT-2 style scaled init on residual projections (attention out-proj and
        # the MLP/SwiGLU down-proj): shrink by 1/sqrt(2*n_layer) so the residual
        # stream variance does not grow with depth.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * config.n_layer) ** 0.5)

        # Weight tying: share the token embedding with the LM head (one matrix,
        # used to embed inputs and to score outputs). Saves vocab*n_embd params.
        if config.tie_weights:
            self.lm_head.weight = self.token_emb.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, t = idx.shape
        if t > self.config.block_size:
            raise ValueError(f"sequence length {t} exceeds block_size {self.config.block_size}")
        x = self.token_emb(idx)  # [B,T,C]
        if self.pos_emb is not None:
            pos = torch.arange(t, device=idx.device)  # [T]
            x = x + self.pos_emb(pos)  # pos broadcasts over batch
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B,T,V]

        loss: torch.Tensor | None = None
        if targets is not None:
            v = logits.size(-1)
            loss = F.cross_entropy(logits.view(-1, v), targets.view(-1))
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
        """Autoregressively extend ``idx`` [B,T], cropping context to block_size."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]  # never exceed the context window
            logits, _ = self(idx_cond)
            next_logits = logits[:, -1, :]  # [B, V]
            next_id = sample_next_token(
                next_logits, temperature=temperature, top_k=top_k, generator=generator
            )
            idx = torch.cat([idx, next_id], dim=1)
        return idx
