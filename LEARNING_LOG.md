# Learning Log

Use this file as a concise engineering notebook.

For every milestone, add:

- Concept and intuition
- Important equation
- Tensor shapes
- Implementation decisions
- Experiment configuration
- Observed result
- Failure or surprising behavior
- What changed in your understanding

Do not write results that were not actually measured.

---

## Stage 1 — Bootstrap (2026-08-03)

**Concept.** Before any model math, set up a reproducible, verifiable substrate:
an installable package, deterministic randomness, and honest reporting of what
hardware we actually have. Every later quality gate (overfit tests, checkpoint
round-trips, causal-mask leakage) depends on being able to fix a seed and pick a
device the same way every run.

**Device policy (`beeja.utils.device`).** Priority MPS → CUDA → CPU. MPS is the
Metal backend on Apple Silicon; CUDA only appears on remote GPU hosts (Colab);
CPU is the universal fallback so tests run anywhere. `select_device()` returns a
`torch.device`.

**Deterministic seeding (`beeja.utils.set_seed`).** One call seeds `PYTHONHASHSEED`,
`random`, NumPy, and PyTorch (CPU/CUDA/MPS). `torch.manual_seed` seeds the CPU
generator (and CUDA if present); MPS needs `torch.mps.manual_seed` separately.
Verified: two `set_seed(1234); torch.rand(8)` calls produce bit-identical tensors;
different seeds differ.

**Measured environment (this machine).**

```json
{"machine": "arm64", "memory_gib": 8.0, "torch": "2.13.0",
 "mps_available": true, "cuda_available": false, "python": "3.13.0"}
```

8 GB unified memory confirmed — this bounds how large a model + optimizer state +
activations we can train locally, so keep local runs to the smoke/Beeja-3M scale.

**Result.** Editable install works; 12 focused tests pass in ~2.2s; ruff clean.

**Understanding shift.** Parameter memory alone is a poor feasibility estimate;
training also needs gradients, optimizer states, activations, and attention
buffers. The 8 GiB figure is the ceiling to plan against, not the model budget.
