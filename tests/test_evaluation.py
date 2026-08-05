"""Evaluation: perplexity identity, diversity, benchmark, report, no state change."""

from __future__ import annotations

import math

import torch

from beeja.data.pipeline import build_datasets
from beeja.evaluation.benchmark import benchmark_generation
from beeja.evaluation.metrics import diversity_report, evaluate_perplexity, ngram_distinct
from beeja.evaluation.report import evaluation_report, render_markdown
from beeja.models.config import smoke_config
from beeja.models.gpt import BeejaGPT
from beeja.utils import set_seed

VOCAB = 40


def _model():
    set_seed(0)
    return BeejaGPT(smoke_config(VOCAB))


def test_perplexity_equals_exp_loss_and_bits():
    model = _model()
    data = torch.randint(VOCAB, (400,))
    m = evaluate_perplexity(
        model,
        data,
        block_size=16,
        batch_size=8,
        batches=10,
        generator=torch.Generator().manual_seed(1),
    )
    assert math.isclose(m["perplexity"], math.exp(m["loss"]), rel_tol=1e-9)
    assert math.isclose(m["bits_per_token"], m["loss"] / math.log(2), rel_tol=1e-9)
    # Random-init model: perplexity is near vocab size.
    assert VOCAB * 0.5 < m["perplexity"] < VOCAB * 2


def test_evaluation_does_not_change_params():
    model = _model()
    data = torch.randint(VOCAB, (400,))
    before = [p.detach().clone() for p in model.parameters()]
    evaluate_perplexity(model, data, block_size=16, batch_size=8, batches=5)
    after = [p.detach().clone() for p in model.parameters()]
    assert all(torch.equal(a, b) for a, b in zip(before, after, strict=True))


def test_ngram_distinct_bounds_and_extremes():
    assert ngram_distinct([1, 1, 1, 1], 1) == 0.25  # one unique token / 4
    assert ngram_distinct([1, 2, 3, 4], 1) == 1.0  # all unique
    assert ngram_distinct([1], 2) == 0.0  # too short for bigrams


def test_diversity_report_keys():
    d = diversity_report([1, 2, 3, 2, 1])
    assert set(d) >= {"distinct_1", "distinct_2", "distinct_3", "repetition_1", "length"}
    assert d["length"] == 5
    assert 0.0 <= d["distinct_2"] <= 1.0


def test_benchmark_returns_positive_throughput():
    model = _model()
    b = benchmark_generation(model, prompt_ids=[1, 2, 3], max_new_tokens=16, device="cpu")
    assert b["tokens_per_second"] > 0
    assert b["params"] > 0
    assert b["device"] == "cpu"


def test_report_has_reproducibility_context_and_renders():
    model = _model()
    _, val_ids, tok = build_datasets("sample", "char", val_fraction=0.1)
    report = evaluation_report(
        model,
        val_ids,
        tok,
        model_config=model.config,
        block_size=16,
        eval_batches=5,
        sample_tokens=20,
        prompts=["the"],
    )
    # Acceptance: config, seed, hardware, and limitations are all present.
    assert "model" in report and "seed" in report
    assert report["hardware"]["machine"]
    assert len(report["limitations"]) >= 1
    md = render_markdown(report, name="Beeja-Test")
    assert "Evaluation Report" in md and "Limitations" in md and "perplexity" in md.lower()
