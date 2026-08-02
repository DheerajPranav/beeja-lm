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

---

## Stage 3 — Beeja-3M decoder-only Transformer (2026-08-03)

**Concept.** Replace the one-step bigram with a stack of pre-norm Transformer
blocks. Each position builds a next-token distribution from *all* earlier
positions via causal self-attention, plus a position-wise MLP.

**Scaled dot-product attention (the core equation).**

    Attention(Q, K, V) = softmax( (Q Kᵀ / √d_k) + causal_mask ) V

The causal mask sets future key positions to `-inf` *before* softmax, so their
probability is exactly 0. Verified two ways: (1) at the attention layer, the
strictly-upper triangle of the attention matrix is all zeros and each row sums
to 1; (2) end-to-end, perturbing the token at position 5 leaves logits at
positions 0–4 bit-unchanged (atol 1e-5) while position 5 changes.

**Tensor shapes (the boundaries).**

    idx            [B, T]
    tok+pos        [B, T, C]                 C = n_embd
    qkv            [B, T, 3C] -> 3 x [B,T,C]
    per-head       [B, n_head, T, head_size] head_size = C / n_head
    att            [B, n_head, T, T]
    context        [B, n_head, T, head_size] -> merge -> [B, T, C]
    logits         [B, T, V]

**Design choices.** Pre-norm residuals (`x = x + sub(LN(x))`) for a clean
identity path; GELU MLP with 4x expansion; GPT-2 init (N(0, 0.02), residual
projections scaled by 1/√(2·n_layer)); weight tying deferred to the modern stage.

**Parameter budget (measured, Beeja-3M, char vocab V=38).**

    total 3,211,776   mlp 2,102,272   attention 1,052,672
    embedding 42,496  lm_head 9,728   norm/other 4,608     (12.25 MiB fp32)

Per block ≈ 12·C² (attention 4C², MLP 8C²); with C=256, L=4 that dominates.
`architecture.md` lists C=128 as a starting point, which measures ~0.8M — so I
tuned C to 256 to make the "Beeja-3M" name honest (~3.2M). Embeddings are a
negligible share because the char vocab is tiny; this will shift once BPE
enlarges the vocab.

**Memory caveat.** Parameter bytes are a floor. Training adds gradients (~1x),
AdamW state (~2x), and activations (∝ batch·T·depth, with attention ∝ T²).

**Result.** Smoke config (2L/d64, 107k params) overfits the sample corpus:
loss 3.67 → 0.26; generated text jumps from bigram gibberish to real words and
phrases ("understanding, not size", "the letters lean on the letters before").
Tiny-batch overfit gate reaches <0.05. Checkpoint round-trip reproduces logits
to atol 1e-6. All on CPU for determinism; MPS is available for larger runs.

**Understanding shift.** Attention is the mechanism that turns "predict from the
previous character" into "predict from the whole visible context" — and the
causal mask is precisely what keeps that prediction honest (no peeking ahead).
