"""Trainer: finite loss/grads, grad accumulation, eval isolation, exact resume."""

from __future__ import annotations

import math
from dataclasses import replace

import torch

from beeja.models.config import smoke_config
from beeja.models.gpt import BeejaGPT
from beeja.training.config import TrainConfig
from beeja.training.trainer import Trainer
from beeja.utils import set_seed

VOCAB = 40


def _data():
    gen = torch.Generator().manual_seed(0)
    train = torch.randint(VOCAB, (400,), generator=gen)
    val = torch.randint(VOCAB, (80,), generator=gen)
    return train, val


def _config(**overrides) -> TrainConfig:
    base = TrainConfig(
        max_steps=12,
        warmup_steps=2,
        batch_size=8,
        block_size=16,
        eval_interval=0,
        checkpoint_interval=0,
        sample_interval=0,
        seed=123,
        device="cpu",
    )
    return replace(base, **overrides)


def _fresh_trainer(config):
    set_seed(0)
    model = BeejaGPT(smoke_config(VOCAB))
    train, val = _data()
    return Trainer(model, train, val, config)


def test_loss_and_grads_are_finite():
    trainer = _fresh_trainer(_config())
    history = trainer.train()
    assert all(math.isfinite(x) for x in history)
    grads = [p.grad for p in trainer.model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_gradient_accumulation_matches_full_batch():
    # K micro-batches of size B (loss/K, summed grads) == one batch of size K*B.
    set_seed(0)
    model = BeejaGPT(smoke_config(VOCAB))
    gen = torch.Generator().manual_seed(1)
    x1 = torch.randint(VOCAB, (4, 16), generator=gen)
    y1 = torch.randint(VOCAB, (4, 16), generator=gen)
    x2 = torch.randint(VOCAB, (4, 16), generator=gen)
    y2 = torch.randint(VOCAB, (4, 16), generator=gen)

    model.zero_grad(set_to_none=True)
    for x, y in ((x1, y1), (x2, y2)):
        _, loss = model(x, y)
        (loss / 2).backward()
    accum = [p.grad.clone() for p in model.parameters()]

    model.zero_grad(set_to_none=True)
    _, loss = model(torch.cat([x1, x2]), torch.cat([y1, y2]))
    loss.backward()
    full = [p.grad.clone() for p in model.parameters()]

    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(accum, full, strict=True))


def test_evaluate_does_not_change_params_or_step():
    trainer = _fresh_trainer(_config())
    trainer.train()
    before = [p.detach().clone() for p in trainer.model.parameters()]
    step_before = trainer.step
    trainer.evaluate()
    after = [p.detach().clone() for p in trainer.model.parameters()]
    assert trainer.step == step_before
    assert all(torch.equal(a, b) for a, b in zip(before, after, strict=True))


def test_resume_reproduces_uninterrupted_run(tmp_path):
    # Uninterrupted: train 12 steps.
    a = _fresh_trainer(_config(max_steps=12))
    a.train()
    reference = [p.detach().clone() for p in a.model.parameters()]

    # Interrupted: train 6, checkpoint, resume in a fresh Trainer, finish to 12.
    b = _fresh_trainer(_config(max_steps=12))
    b.train(until_step=6)
    ckpt = tmp_path / "resume.pt"
    b.save(ckpt)

    train, val = _data()
    c = Trainer.resume(ckpt, train, val)
    assert c.step == 6
    c.train()

    assert c.step == 12
    resumed = [p.detach().clone() for p in c.model.parameters()]
    assert all(torch.allclose(r, s, atol=1e-6) for r, s in zip(reference, resumed, strict=True))


def test_gradient_clipping_config_runs():
    trainer = _fresh_trainer(_config(grad_clip=0.5))
    history = trainer.train()
    assert len(history) == 12
