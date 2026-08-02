# Testing and Verification Guide

## General tests

- Fixed-seed determinism for small CPU tests.
- Explicit shape and dtype checks.
- Empty input and invalid configuration errors.
- Device movement tests where practical.
- Save/load round trips.

## Tokenizer tests

- ASCII encode/decode round trip.
- Unicode encode/decode round trip.
- Empty string behavior.
- Unknown/special-token behavior.
- Deterministic training with identical input.
- Save/load identity.

## Attention tests

- Output shape equals input sequence shape in model dimension.
- Attention probabilities sum to one over allowed keys.
- Future positions have zero probability.
- Perturbing a future token cannot alter earlier logits in evaluation mode.
- Invalid head divisibility fails clearly.

## Training tests

- One optimization step changes at least one parameter.
- Loss is finite.
- Gradients are finite.
- Tiny-batch overfit test.
- Gradient accumulation matches the intended effective batch behavior within tolerance.
- Scheduler state survives checkpoint resume.

## Generation tests

- Greedy decoding is deterministic.
- Temperature must be positive.
- Top-k bounds are validated.
- EOS or stop token terminates generation.
- KV-cached and uncached generation agree for a short fixed sequence.

## Data leakage checks

- Train and validation examples are separated before batching.
- Sequential documents do not cross boundaries unless explicitly allowed.
- Evaluation does not update model or optimizer state.

## Stage completion rule

A stage is complete only when:

1. Its focused tests pass.
2. Relevant broader tests pass.
3. A smoke execution succeeds.
4. Actual command output is summarized.
5. `PROJECT_STATE.md` contains acceptance evidence.
