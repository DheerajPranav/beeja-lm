"""Modern components: RMSNorm, RoPE, SwiGLU, weight tying, AMP, integration."""

from __future__ import annotations

import torch

from beeja.models.config import beeja_3m_config, beeja_3m_modern_config, modern_smoke_config
from beeja.models.gpt import BeejaGPT
from beeja.models.rmsnorm import RMSNorm
from beeja.models.rope import apply_rope, build_rope_cache
from beeja.models.swiglu import SwiGLU, swiglu_hidden
from beeja.training.basic import fit_batch
from beeja.utils import parameter_count, set_seed

VOCAB = 40


# -- RMSNorm ----------------------------------------------------------------
def test_rmsnorm_shape_and_unit_rms():
    norm = RMSNorm(16)
    x = torch.randn(4, 8, 16) * 5.0
    out = norm(x)
    assert out.shape == x.shape
    # With weight initialised to ones, each row has unit root-mean-square.
    rms = out.pow(2).mean(dim=-1)
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)


def test_rmsnorm_matches_manual_formula():
    norm = RMSNorm(8)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.5, 2.0, 8))
    x = torch.randn(3, 8)
    manual = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.eps) * norm.weight
    assert torch.allclose(norm(x), manual, atol=1e-6)


# -- RoPE -------------------------------------------------------------------
def test_rope_preserves_shape_and_pos0_is_identity():
    cos, sin = build_rope_cache(seq_len=8, head_size=16)
    x = torch.randn(2, 4, 8, 16)  # [B, n_head, T, head_size]
    out = apply_rope(x, cos, sin)
    assert out.shape == x.shape
    # Position 0 rotates by angle 0 -> identity.
    assert torch.allclose(out[:, :, 0, :], x[:, :, 0, :], atol=1e-6)


def test_rope_dot_product_depends_only_on_relative_position():
    head_size = 16
    cos, sin = build_rope_cache(seq_len=10, head_size=head_size)
    q = torch.randn(1, 1, 10, head_size)
    k = torch.randn(1, 1, 10, head_size)
    assert apply_rope(q, cos, sin).shape == q.shape  # shape preserved

    def score(m, n):  # <RoPE(q_m), RoPE(k_n)> using the SAME base vectors
        qm = apply_rope(q[:, :, :1, :], cos[m : m + 1], sin[m : m + 1])
        kn = apply_rope(k[:, :, :1, :], cos[n : n + 1], sin[n : n + 1])
        return (qm * kn).sum().item()

    # Same relative distance (-3) at different absolute positions -> equal score.
    assert abs(score(2, 5) - score(4, 7)) < 1e-4


# -- SwiGLU -----------------------------------------------------------------
def test_swiglu_shape_and_hidden_sizing():
    cfg = modern_smoke_config(VOCAB)
    mlp = SwiGLU(cfg)
    x = torch.randn(2, 5, cfg.n_embd)
    assert mlp(x).shape == (2, 5, cfg.n_embd)
    assert swiglu_hidden(cfg.n_embd) % 8 == 0


# -- weight tying -----------------------------------------------------------
def test_weight_tying_shares_matrix_and_saves_params():
    set_seed(0)
    tied = BeejaGPT(beeja_3m_modern_config(VOCAB))
    assert tied.lm_head.weight is tied.token_emb.weight  # same tensor object

    set_seed(0)
    untied = BeejaGPT(beeja_3m_config(VOCAB))  # learned/gelu/layernorm, untied
    # Tying removes one vocab x n_embd matrix from the count.
    saved = untied.config.vocab_size * untied.config.n_embd
    assert parameter_count(untied)["lm_head"] == saved
    assert parameter_count(tied)["lm_head"] == 0


# -- integration ------------------------------------------------------------
def test_modern_model_forward_and_overfit():
    set_seed(0)
    model = BeejaGPT(modern_smoke_config(VOCAB))
    idx = torch.randint(VOCAB, (4, 16))
    logits, loss = model(idx, idx)
    assert logits.shape == (4, 16, VOCAB)

    gen = torch.Generator().manual_seed(0)
    x = torch.randint(VOCAB, (4, 16), generator=gen)
    y = torch.randint(VOCAB, (4, 16), generator=gen)
    losses = fit_batch(model, x, y, steps=300, lr=3e-3)
    assert losses[-1] < 0.1, f"modern model failed to overfit: {losses[-1]:.4f}"


def test_modern_model_has_no_future_leakage():
    set_seed(0)
    model = BeejaGPT(modern_smoke_config(VOCAB)).eval()
    idx = torch.randint(VOCAB, (1, 8))
    a, _ = model(idx)
    idx2 = idx.clone()
    idx2[0, 5] = (idx2[0, 5] + 1) % VOCAB
    b, _ = model(idx2)
    # RoPE must not break causality: logits before position 5 stay identical.
    assert torch.allclose(a[:, :5, :], b[:, :5, :], atol=1e-5)
    assert not torch.allclose(a[:, 5, :], b[:, 5, :])


def test_amp_training_step_is_finite():
    import math

    from beeja.training.config import TrainConfig
    from beeja.training.trainer import Trainer

    set_seed(0)
    model = BeejaGPT(modern_smoke_config(VOCAB))
    data = torch.randint(VOCAB, (400,))
    cfg = TrainConfig(
        max_steps=3,
        warmup_steps=1,
        batch_size=8,
        block_size=16,
        eval_interval=0,
        checkpoint_interval=0,
        amp=True,
        amp_dtype="bf16",
        device="cpu",
    )
    trainer = Trainer(model, data, data, cfg)
    history = trainer.train()
    assert all(math.isfinite(x) for x in history)
