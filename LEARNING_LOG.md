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

---

## Stage 4 — Byte-level BPE tokenizer (2026-08-03)

**Concept.** Character tokens are simple but wasteful: one token per character
means long sequences and no shared structure between "seed" and "seeds".
Byte-Pair Encoding learns a middle ground — start from raw bytes and repeatedly
fuse the most frequent adjacent pair into a new token.

**Why byte-level.** UTF-8 encode first, so the base vocabulary is exactly the
256 byte values. Any Unicode string (accents, CJK, emoji) is then representable
with **no unknown token**, and decode is exact. Verified round trips on
`"café — bīja 種子 🌱🌳 naïve"` — text never seen during training.

**Algorithm.**
- Train: `ids = utf8(text)`; loop `num_merges = vocab_size - 256` times: count
  adjacent pairs (`get_stats`), pick the most frequent, `merge` it into id
  `256+i`, record the merge. Determinism: `max(stats, key=stats.get)` — ties
  broken by first appearance via stable dict order, so identical input ⇒
  identical merges (tested).
- Encode: greedily merge the present pair with the **lowest learned rank** until
  no learned pair remains.
- Decode: concatenate each id's byte string, UTF-8 decode.

**Special tokens.** Registered above the BPE range (e.g. `<|endoftext|>` → 320).
Recognised only when `allowed_special` permits; otherwise the literal string is
encoded as ordinary bytes. This prevents user text from spoofing control tokens.

**Save/load.** JSON stores the ordered merges + special tokens; the byte vocab is
rebuilt deterministically because every merge combines lower ids into a higher
one. Reload is identical (merges, vocab, and encodings all match).

**Measured compression (held-out val text, 64 merges).**

    bytes=64  tokens=26  bytes_per_token=2.46

So each token stands in for ~2.46 bytes vs the byte-level baseline of 1.0. The
corpus is tiny, so this is a mechanism demonstration, not a production ratio.

**Understanding shift.** Tokenization is a *compression* choice that trades
vocabulary size against sequence length. More merges → shorter sequences (cheaper
attention, which is O(T²)) but a larger embedding/LM-head. The next stages can
now train on subword ids instead of characters.

---

## Stage 5 — Resumable pretraining pipeline (2026-08-03)

**Concept.** Turn the one-off training loop into infrastructure: a config-driven,
resumable trainer with a learning-rate schedule, gradient accumulation, gradient
clipping, leakage-safe validation, checkpointing, and sample generation.

**LR schedule.** Linear warmup then cosine decay:

    step < warmup:  lr = max_lr · (step+1)/warmup
    else:           lr = min_lr + ½(1+cos(π·ratio))·(max_lr−min_lr),
                    ratio = (step−warmup)/(max_steps−warmup)

Warmup protects a fresh model from a destabilising first step; cosine anneals to
small refining steps. Verified: ramps to max_lr at the warmup boundary, decays
monotonically, floors at min_lr past max_steps.

**Gradient accumulation (the key identity).** K micro-batches of size B, each
back-propagating `loss/K`, sum to exactly the gradient of one batch of size K·B
— because cross-entropy averages over tokens and every micro-batch has the same
token count. Tested to atol 1e-5. This is how a small machine simulates a large
effective batch (here B·grad_accum).

**Leakage discipline (a bug the tests caught).** First cut trained the char
tokenizer on the train split only; a val-only character ('k') then had no id.
Fix encodes the real distinction: the **alphabet / byte base** is the symbol
space (built from the full corpus — not leakage), while **BPE merges** are
*learned statistics* (train-only). Validation also runs under no-grad with a
fixed generator and never touches optimizer/params (tested).

**Exact resume.** The checkpoint stores model + optimizer state + step + the
batch-sampling generator's RNG state. Restoring all four means an interrupted run
continues on the identical trajectory. Verified two ways: a unit test (interrupted
vs uninterrupted final params match to atol 1e-6) and the CLI (resume from step
100 reproduced step-200 loss 0.1988 / val 5.1661 exactly).

**Result.** Smoke config (2L/d64, block 16, BPE vocab 320, 143,104 params) runs
200 steps in seconds on CPU: train loss → ~0.20 while val rises to ~5.1. That gap
is honest overfitting of a ~640-char corpus — the pipeline is correct; the data
is deliberately tiny. The Beeja-10M full-run config + a reproducible dataset
download script are prepared but **not** launched (per project policy).

**Understanding shift.** "Training a model" is mostly bookkeeping around the loss:
schedule, accumulation, clipping, evaluation hygiene, and — most underrated —
being able to stop and resume without changing the result. Determinism is a
feature you engineer (own your RNG state), not something you hope for.

---

## Stage 6 — Modern architecture (2026-08-04)

**Concept.** Upgrade the baseline block to the components a 2024-era LLM uses,
each added as a config switch on the same `BeejaGPT` so a baseline and a modern
model differ only in these parts (defaults keep the original baseline).

**RoPE (Rotary Position Embeddings).** Rotate Q and K by an angle ∝ position
instead of adding a learned vector. Key property (and the test): the score
⟨R_m·q, R_n·k⟩ depends only on the *relative* distance m−n — verified numerically
that score(2,5) == score(4,7). Position 0 is the identity rotation. Implemented
with the "rotate-half" layout: `x·cos + rotate_half(x)·sin`, cos/sin of shape
[T, head_size]. RoPE lives *inside* attention, so the model drops the learned
position embedding entirely — yet stays strictly causal (leakage test still passes).

**RMSNorm.** Drop LayerNorm's mean-centring; rescale by the root-mean-square:
`x / sqrt(mean(x²)+eps) · weight`. No bias, half the norm parameters, and the
statistic is computed in float32 for mixed-precision stability. Verified against
the manual formula and that output rows have unit RMS.

**SwiGLU.** Gated feed-forward: `(SiLU(x·W_gate) ⊙ (x·W_up))·W_down`. Three
matrices with hidden ≈ 8d/3 so the parameter budget stays ~equal to the 4×
GELU MLP (8·d²).

**Weight tying.** Share one matrix between the token embedding and the LM head.
`self.lm_head.weight = self.token_emb.weight` — the same tensor object embeds
inputs and scores outputs, removing a whole vocab×d matrix from the count.

**Mixed precision.** `torch.autocast` wraps only the forward pass (bf16 by
default; fp16+GradScaler left for the CUDA full run). One step runs finite on CPU
bf16 in the test.

**Controlled comparison (same seed/data/schedule, tiny corpus).**

    arch        params     MiB   loss0    lossN     val
    baseline   142,080    0.54   5.756    0.114    5.314
    modern     122,496    0.47   5.766    0.123    5.257

Modern is *leaner* (RoPE removes position emb, tying removes the head, RMSNorm
halves norm params) at ~equal capacity. Val is comparable — on a ~640-char corpus
this compares the machinery, not scaled quality. Modern Beeja-3M measures
3,184,768 params (vs baseline 3,211,776), still ~3M.

**Understanding shift.** The modern stack is less about "more" and more about
*better-conditioned*: relative positions that extrapolate, a cheaper norm, a
gated MLP, and a shared embedding — most of which *reduce* parameters while
improving trainability. The next real training run should use these flags.
