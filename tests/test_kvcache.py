"""KV cache: cached and uncached generation agree, for baseline and modern."""

from __future__ import annotations

import torch

from beeja.models.config import modern_smoke_config, smoke_config
from beeja.models.gpt import BeejaGPT
from beeja.utils import set_seed

VOCAB = 40


def _agree(config):
    set_seed(0)
    model = BeejaGPT(config).eval()
    prompt = torch.randint(VOCAB, (1, 5))
    # Greedy (top_k=1) is deterministic, so identical logits -> identical tokens.
    uncached = model.generate(prompt, 30, top_k=1, use_cache=False)
    cached = model.generate(prompt, 30, top_k=1, use_cache=True)
    assert torch.equal(uncached, cached)


def test_cached_matches_uncached_baseline():
    _agree(smoke_config(VOCAB))  # learned positions + LayerNorm + GELU


def test_cached_matches_uncached_modern():
    _agree(modern_smoke_config(VOCAB))  # RoPE + RMSNorm + SwiGLU (RoPE offset path)


def test_cache_still_agrees_past_block_size():
    # Generate beyond block_size so the cache-refill/crop path is exercised.
    cfg = smoke_config(VOCAB)  # block_size 32
    set_seed(0)
    model = BeejaGPT(cfg).eval()
    prompt = torch.randint(VOCAB, (1, 4))
    n = cfg.block_size + 10
    assert torch.equal(
        model.generate(prompt, n, top_k=1, use_cache=False),
        model.generate(prompt, n, top_k=1, use_cache=True),
    )


def test_generate_stream_yields_expected_count():
    set_seed(0)
    model = BeejaGPT(smoke_config(VOCAB)).eval()
    prompt = torch.randint(VOCAB, (1, 3))
    toks = list(model.generate_stream(prompt, 12, top_k=1))
    assert len(toks) == 12
    assert all(0 <= int(t.item()) < VOCAB for t in toks)
