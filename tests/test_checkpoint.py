"""Checkpoint round-trip: reloaded model reproduces logits and keeps its name."""

from __future__ import annotations

import torch

from beeja.models.config import smoke_config
from beeja.models.gpt import BeejaGPT
from beeja.training.checkpoint import load_checkpoint, save_checkpoint
from beeja.utils import set_seed


def test_save_load_preserves_logits_and_metadata(tmp_path):
    set_seed(0)
    model = BeejaGPT(smoke_config(40))
    model.eval()
    idx = torch.randint(40, (2, 12))
    logits_before, _ = model(idx)

    path = tmp_path / "beeja-3m.pt"
    save_checkpoint(path, model, name="Beeja-3M", step=123, extra={"note": "test"})

    reloaded, ckpt = load_checkpoint(path)
    logits_after, _ = reloaded(idx)

    assert torch.allclose(logits_before, logits_after, atol=1e-6)
    assert ckpt["name"] == "Beeja-3M"
    assert ckpt["step"] == 123
    assert ckpt["config"]["n_embd"] == 64


def test_reloaded_config_rebuilds_same_architecture(tmp_path):
    set_seed(1)
    model = BeejaGPT(smoke_config(40))
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, name="Beeja-3M")
    reloaded, _ = load_checkpoint(path)
    assert reloaded.config == model.config
