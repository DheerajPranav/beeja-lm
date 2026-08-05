"""Evaluation: intrinsic metrics, runtime benchmarks, and reproducible reports."""

from __future__ import annotations

from beeja.evaluation.benchmark import benchmark_generation
from beeja.evaluation.metrics import diversity_report, evaluate_perplexity, ngram_distinct
from beeja.evaluation.report import evaluation_report, render_markdown

__all__ = [
    "evaluate_perplexity",
    "ngram_distinct",
    "diversity_report",
    "benchmark_generation",
    "evaluation_report",
    "render_markdown",
]
