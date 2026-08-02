"""Seeding makes Python, NumPy, and PyTorch RNGs reproducible."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from beeja.utils import set_seed


def test_torch_rng_is_reproducible():
    set_seed(1234)
    a = torch.rand(8)
    set_seed(1234)
    b = torch.rand(8)
    assert torch.equal(a, b)


def test_numpy_and_python_rng_are_reproducible():
    set_seed(7)
    np_a, py_a = np.random.rand(4).tolist(), [random.random() for _ in range(4)]
    set_seed(7)
    np_b, py_b = np.random.rand(4).tolist(), [random.random() for _ in range(4)]
    assert np_a == np_b
    assert py_a == py_b


def test_different_seeds_differ():
    set_seed(1)
    a = torch.rand(8)
    set_seed(2)
    b = torch.rand(8)
    assert not torch.equal(a, b)


def test_negative_seed_rejected():
    with pytest.raises(ValueError):
        set_seed(-1)
