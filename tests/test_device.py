"""Device selection returns a usable device and respects the preference order."""

from __future__ import annotations

import pytest
import torch

from beeja.utils import select_device


def test_returns_torch_device_from_allowed_set():
    device = select_device()
    assert isinstance(device, torch.device)
    assert device.type in {"mps", "cuda", "cpu"}


def test_cpu_only_preference_yields_cpu():
    assert select_device(prefer=("cpu",)).type == "cpu"


def test_empty_preference_falls_back_to_cpu():
    assert select_device(prefer=()).type == "cpu"


def test_unknown_backend_name_raises():
    with pytest.raises(ValueError):
        select_device(prefer=("tpu",))
