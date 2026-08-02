"""Parameter counting: the breakdown sums to the total; Beeja-3M lands near 3M."""

from __future__ import annotations

from beeja.models.config import beeja_3m_config, smoke_config
from beeja.models.gpt import BeejaGPT
from beeja.utils import parameter_count


def test_breakdown_sums_to_total():
    model = BeejaGPT(smoke_config(50))
    counts = parameter_count(model)
    parts = (
        counts["embedding"]
        + counts["attention"]
        + counts["mlp"]
        + counts["lm_head"]
        + counts["norm_other"]
    )
    assert parts == counts["total"]
    assert counts["trainable"] == counts["total"]  # nothing frozen


def test_beeja_3m_is_close_to_three_million():
    model = BeejaGPT(beeja_3m_config(vocab_size=38))
    counts = parameter_count(model)
    # "Close to 3M": accept a generous band around the target.
    assert 2_500_000 <= counts["total"] <= 3_500_000, counts["total"]
    # Most parameters live in attention + MLP, not embeddings (small char vocab).
    assert counts["mlp"] > counts["embedding"]
