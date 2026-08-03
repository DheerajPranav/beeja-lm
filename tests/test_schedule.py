"""Learning-rate schedule: warmup ramp, cosine decay, and floor."""

from __future__ import annotations

from beeja.training.schedule import lr_at

KW = dict(warmup_steps=10, max_steps=100, max_lr=1.0, min_lr=0.1)


def test_warmup_ramps_up_to_max():
    assert lr_at(0, **KW) < lr_at(5, **KW) < lr_at(9, **KW)
    assert abs(lr_at(9, **KW) - 1.0) < 1e-9  # last warmup step hits max_lr
    assert abs(lr_at(10, **KW) - 1.0) < 1e-9  # cosine starts at max_lr (ratio 0)


def test_cosine_decays_monotonically():
    decay = [lr_at(s, **KW) for s in range(10, 101, 10)]
    assert decay == sorted(decay, reverse=True)  # strictly non-increasing


def test_reaches_min_lr_floor():
    assert abs(lr_at(100, **KW) - 0.1) < 1e-9
    assert abs(lr_at(500, **KW) - 0.1) < 1e-9  # stays at floor past the end


def test_midpoint_is_between_min_and_max():
    mid = lr_at(55, **KW)  # ~halfway through decay
    assert 0.1 < mid < 1.0


def test_zero_warmup_starts_at_cosine():
    kw = dict(warmup_steps=0, max_steps=100, max_lr=1.0, min_lr=0.1)
    assert abs(lr_at(0, **kw) - 1.0) < 1e-9
