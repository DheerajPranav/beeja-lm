"""Runtime benchmark: generation latency, throughput, memory, and model size.

Measures wall-clock generation speed (tokens/second) after a short warm-up, plus
parameter count and — on CUDA — peak memory. These are the practical numbers that
decide whether a model is usable for local inference.
"""

from __future__ import annotations

import time
from typing import Any

import torch

from beeja.utils import parameter_count


@torch.no_grad()
def benchmark_generation(
    model: torch.nn.Module,
    *,
    prompt_ids: list[int],
    max_new_tokens: int = 128,
    device: torch.device | str = "cpu",
    runs: int = 1,
) -> dict[str, Any]:
    device = str(device)
    model.eval()
    model.to(device)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    model.generate(idx, 4)  # warm-up (kernels, caches)
    is_cuda = device.startswith("cuda")
    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    for _ in range(runs):
        model.generate(idx, max_new_tokens)
    if is_cuda:
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / runs

    counts = parameter_count(model)
    result: dict[str, Any] = {
        "tokens_generated": max_new_tokens,
        "seconds": round(elapsed, 4),
        "tokens_per_second": round(max_new_tokens / elapsed, 2) if elapsed > 0 else float("inf"),
        "params": counts["total"],
        "param_mib": counts["param_mib"],
        "device": device,
    }
    if is_cuda:
        result["peak_memory_mib"] = round(torch.cuda.max_memory_allocated() / 1024**2, 2)
    return result
