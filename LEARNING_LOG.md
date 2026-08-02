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

---

## Stage 2 — Bigram baseline (2026-08-03)

**Concept.** A bigram model predicts the next character from *only* the current
character. It is a single lookup table `W` of shape `[V, V]`: row `i` is the
logit vector for whatever follows token `i`. There is no context beyond one
step and no learned representation — the point is to establish the floor that
self-attention must beat.

**Objective.** Next-token cross-entropy:

    L = -(1/T) * sum_t log p(x_t | x_{t-1})

At random init every next-token distribution is roughly uniform, so loss starts
near `ln(V)`. Here `V = 38`, so `ln(38) ≈ 3.64`; measured first-step loss ~4.07
(a touch above uniform due to init variance) and it fell to ~2.06.

**Tensor shapes (the boundaries that matter).**

    idx      [B, T]        input character ids
    W = Embedding(V, V)
    logits   [B, T, V]     one distribution per position (just W[idx])
    targets  [B, T]        next-character ids
    for cross_entropy: logits -> [B*T, V], targets -> [B*T]

Batching: from a 1-D id stream of length N, sample B random start offsets, take
windows of length `block_size`; `y` is `x` shifted by one (`x[:,1:] == y[:,:-1]`).
Train/val split is **positional** (val = tail) so no window straddles the
boundary — this is the data-leakage guard.

**Sampling controls.** `logits / temperature` then softmax; top-k masks all but
the k largest logits to `-inf` before softmax. Verified: `top_k=1` reduces to
argmax; a fixed `torch.Generator` makes both batching and generation
bit-reproducible.

**Result.** `params = V² = 1444`. Loss 4.07 → 2.06 over 500 steps (val 2.17).
Samples reproduce local character statistics (spaces after words, "seed,",
q→u tendencies) but no word/sentence structure — the expected bigram ceiling.

**Understanding shift.** The gap between a bigram (context = 1) and language
that needs long-range dependencies is exactly the motivation for causal
self-attention in the next stage: give each position access to *all* earlier
positions, not just the previous one.
