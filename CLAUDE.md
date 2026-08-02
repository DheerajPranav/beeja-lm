# Beeja LM Project Instructions

## Identity

- Model family: `Beeja`.
- First named checkpoint: `Beeja-3M`.
- Repository/package identity: `beeja-lm` / `beeja`.
- Future checkpoints should use consistent names such as `Beeja-10M`, `Beeja-30M`, and `Beeja-3M-Instruct`.

## Mission

Build Beeja-3M, an educational decoder-only language model from random initialization, and evolve it into a clearly versioned model family. The repository must expose each important mechanism clearly enough that the learner can explain, test, and modify it.

## Source of truth

- Read `PROJECT_STATE.md` before changing code.
- Use `.claude/skills/beeja-lab/references/roadmap.md` for stage scope.
- Use `.claude/skills/beeja-lab/references/architecture.md` for model targets.
- Use `.claude/skills/beeja-lab/references/testing.md` for quality gates.
- Record completed work and commands in `PROJECT_STATE.md`.
- Record concepts, equations, experiments, and observations in `LEARNING_LOG.md`.

## Hardware constraints

- Primary machine: Apple Silicon M2 with 8 GB unified memory.
- Primary implementation: Python and PyTorch.
- Local device order: MPS, then CPU. Never assume CUDA locally.
- Use Google Colab only for larger training runs or experiments.
- Keep local defaults small and safe.
- Never launch a long or expensive training run without explicit user approval.
- A normal implementation turn may run smoke tests of a few minutes, not full pretraining.

## Development workflow

1. Inspect the repository and current state.
2. State the exact milestone and acceptance criteria.
3. Implement the smallest complete vertical slice.
4. Add or update tests with the implementation.
5. Run formatting, linting, unit tests, and relevant smoke tests.
6. Explain failures and fix them rather than bypassing them.
7. Update `PROJECT_STATE.md` and `LEARNING_LOG.md`.
8. Finish with changed files, commands run, results, and the next milestone.

## Educational requirements

- Do not hide core Transformer logic behind high-level model libraries.
- PyTorch tensor operations and `torch.nn` building blocks are allowed.
- Do not use Hugging Face model classes to implement the core decoder-only model.
- Hugging Face datasets/tokenizers may be used later only as comparison tools, not as the primary implementation.
- Explain tensor shapes at important boundaries.
- Include the mathematical equation near major mechanisms where useful.
- Prefer readable code over premature optimization.
- Every major component needs a focused test.

## Architecture boundaries

- Decoder-only autoregressive Transformer.
- Causal next-token prediction objective.
- Begin with character tokenization and a bigram baseline.
- Implement causal self-attention manually.
- Build a custom byte-level BPE tokenizer.
- Add RoPE, RMSNorm, SwiGLU, weight tying, mixed precision, and KV caching in later stages.
- Add supervised instruction tuning only after base pretraining works.

## Required quality gates

- Deterministic seeds where practical.
- Shape and dtype assertions around attention and batching.
- Causal-mask leakage test.
- Tiny-batch overfit test.
- Save/load checkpoint round-trip test.
- Encode/decode tokenizer round-trip test.
- Generation smoke test.
- No silent exception swallowing.
- No fabricated metrics or claims of successful training.

## Project layout target

```text
src/beeja/
  data/
  tokenizer/
  models/
  training/
  generation/
  evaluation/
  utils/
tests/
configs/
notebooks/
scripts/
reports/
checkpoints/
```

## Commands

Prefer a small, consistent command surface through `pyproject.toml`, `Makefile`, or scripts:

```bash
python -m pytest
python -m beeja.train --config configs/smoke.yaml
python -m beeja.generate --checkpoint <path> --prompt "..."
```

Keep generated datasets, checkpoints, caches, and secrets out of Git.
