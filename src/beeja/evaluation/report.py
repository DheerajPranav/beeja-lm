"""Assemble a reproducible evaluation report.

A report bundles quality metrics (perplexity), generation diversity, a runtime
benchmark, sample continuations for a small prompt set, and the full context
needed to reproduce it: model config, seed, hardware, and stated limitations.

No LLM-as-judge is used anywhere — metrics are intrinsic (loss/perplexity) and
lexical (distinct-n), plus wall-clock speed.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch

from beeja.evaluation.benchmark import benchmark_generation
from beeja.evaluation.metrics import diversity_report, evaluate_perplexity
from beeja.utils import environment_report

DEFAULT_PROMPTS = ["Once upon a time", "The little", "One day"]

LIMITATIONS = [
    "Perplexity is dataset-relative; compare only within the same tokenizer and corpus.",
    "Diversity is lexical (distinct-n); it does not measure coherence or truthfulness.",
    "A ~3M-parameter model produces simple text; do not read fluency as understanding.",
    "No held-out human or LLM judgement is included.",
]


@torch.no_grad()
def evaluation_report(
    model: torch.nn.Module,
    val_data: torch.Tensor,
    tokenizer: Any,
    *,
    model_config: Any,
    block_size: int,
    batch_size: int = 32,
    eval_batches: int = 50,
    seed: int = 1337,
    device: torch.device | str = "cpu",
    prompts: list[str] | None = None,
    sample_tokens: int = 120,
) -> dict[str, Any]:
    prompts = prompts if prompts is not None else DEFAULT_PROMPTS
    gen = torch.Generator().manual_seed(seed)

    perplexity = evaluate_perplexity(
        model,
        val_data,
        block_size=block_size,
        batch_size=batch_size,
        batches=eval_batches,
        device=device,
        generator=gen,
    )

    samples = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt) if prompt else [0]
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(idx, sample_tokens, temperature=0.8)[0].tolist()
        samples.append(
            {"prompt": prompt, "text": tokenizer.decode(out), "diversity": diversity_report(out)}
        )

    bench = benchmark_generation(
        model,
        prompt_ids=tokenizer.encode(prompts[0]) or [0],
        max_new_tokens=sample_tokens,
        device=device,
    )

    return {
        "model": asdict(model_config),
        "seed": seed,
        "metrics": perplexity,
        "benchmark": bench,
        "samples": samples,
        "hardware": environment_report(),
        "limitations": LIMITATIONS,
    }


def render_markdown(report: dict[str, Any], name: str = "Beeja") -> str:
    m, b = report["metrics"], report["benchmark"]
    lines = [
        f"# {name} — Evaluation Report",
        "",
        f"- seed: `{report['seed']}`  ·  device: `{b['device']}`  ·  "
        f"params: `{b['params']:,}` ({b['param_mib']} MiB)",
        f"- hardware: `{report['hardware'].get('machine')}` · "
        f"torch `{report['hardware'].get('torch')}`",
        "",
        "## Quality",
        "",
        "| metric | value |",
        "|---|---|",
        f"| val loss | {m['loss']:.4f} |",
        f"| perplexity | {m['perplexity']:.3f} |",
        f"| bits/token | {m['bits_per_token']:.4f} |",
        "",
        "## Runtime",
        "",
        "| metric | value |",
        "|---|---|",
        f"| tokens/second | {b['tokens_per_second']} |",
        f"| seconds ({b['tokens_generated']} tok) | {b['seconds']} |",
    ]
    if "peak_memory_mib" in b:
        lines.append(f"| peak memory (MiB) | {b['peak_memory_mib']} |")
    lines += ["", "## Sample continuations", ""]
    for s in report["samples"]:
        d = s["diversity"]
        lines += [
            f"**Prompt:** `{s['prompt']}`  ·  distinct-2 = {d['distinct_2']}",
            "",
            "```",
            s["text"].strip(),
            "```",
            "",
        ]
    lines += ["## Limitations", ""]
    lines += [f"- {item}" for item in report["limitations"]]
    return "\n".join(lines) + "\n"
