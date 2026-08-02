# Architecture Targets

Treat these as starting points. Measure memory and throughput before increasing size.

## Smoke model

```text
layers: 2
model dimension: 64
heads: 4
context: 64
vocabulary: character or tiny BPE
purpose: unit tests and pipeline smoke runs
```

## Beeja-3M — first named release

```text
layers: 4
model dimension: 128
heads: 4
context: 128
target scale: approximately 3M parameters
release name: Beeja-3M
purpose: local M2 training, architecture learning, and the first publishable checkpoint
```

## Intermediate model

```text
layers: 6
model dimension: 256
heads: 8
context: 256
vocabulary: 4K custom BPE
expected scale: roughly 8–15M parameters
purpose: Colab pretraining and local inference
```

## Final educational model

```text
layers: 8–10
model dimension: 384–512
heads: 6–8
context: 512
vocabulary: 8K–16K custom BPE
expected scale: roughly 20–50M parameters
purpose: modern architecture, instruction tuning, local inference
```

## Core equations

### Next-token objective

For token sequence `x_1, ..., x_T`, minimize:

```text
L = -(1/T) * sum_t log p(x_t | x_<t)
```

### Scaled dot-product attention

```text
Attention(Q, K, V) = softmax((Q K^T / sqrt(d_k)) + causal_mask) V
```

The causal mask must make every future position unavailable before softmax.

### Residual block

Use an explicit, testable pre-normalization or post-normalization design. Document the decision and keep it consistent.

## Parameter counting

Implement a utility that reports:

- Total parameters.
- Trainable parameters.
- Embedding parameters.
- Attention parameters.
- MLP parameters.
- Approximate parameter memory by dtype.

Do not estimate final feasibility using parameter memory alone. Training also needs gradients, optimizer states, activations, attention matrices, and temporary buffers.

## Naming convention

- `Beeja-3M`: first compact baseline release.
- `Beeja-10M`: intermediate BPE/pretraining release.
- `Beeja-30M`: modern architecture release.
- Append `-Instruct` for supervised chat tuning and `-Base` when distinguishing base checkpoints.
- Parameter suffixes describe approximate parameter counts and must be replaced by measured counts in model cards.
