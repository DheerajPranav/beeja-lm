"""Inference helpers: checkpoint loading errors and streaming completion."""

from __future__ import annotations

import pytest
import torch

from beeja.data.char import CharTokenizer
from beeja.inference import load_model, stream_completion
from beeja.models.config import smoke_config
from beeja.models.gpt import BeejaGPT
from beeja.training.checkpoint import save_checkpoint
from beeja.utils import set_seed

# A char tokenizer + a model sized to its vocab (as the app always pairs them).
TOK = CharTokenizer.from_text("hello world, the seed grows.")


def _save_smoke_checkpoint(path, vocab: int = 40):
    set_seed(0)
    model = BeejaGPT(smoke_config(vocab))
    save_checkpoint(path, model, name="Beeja-Test")
    return model


def test_load_model_round_trips(tmp_path):
    path = tmp_path / "m.pt"
    _save_smoke_checkpoint(path)
    model, ckpt = load_model(path, device="cpu")
    assert ckpt["name"] == "Beeja-Test"
    assert model.config.vocab_size == 40


def test_load_missing_checkpoint_raises_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "does-not-exist.pt")


def test_load_non_beeja_checkpoint_raises(tmp_path):
    bad = tmp_path / "bad.pt"
    torch.save({"just": "a dict"}, bad)
    with pytest.raises(ValueError):
        load_model(bad)


def test_stream_completion_yields_text(tmp_path):
    path = tmp_path / "m.pt"
    _save_smoke_checkpoint(path, vocab=TOK.vocab_size)
    model, _ = load_model(path)
    deltas = list(
        stream_completion(model, TOK, "the seed", max_new_tokens=10, top_k=1, device="cpu")
    )
    assert isinstance("".join(deltas), str)
    assert 0 < len(deltas) <= 10


def test_stream_completion_caps_and_counts(tmp_path):
    path = tmp_path / "m.pt"
    _save_smoke_checkpoint(path, vocab=TOK.vocab_size)
    model, _ = load_model(path)
    deltas = list(stream_completion(model, TOK, "the", max_new_tokens=5, top_k=1))
    assert len(deltas) == 5  # one delta per generated token
