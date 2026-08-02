# Beeja Project State

## Model identity

- Family: `Beeja`
- Current target: `Beeja-3M`
- Repository: `beeja-lm`
- Python package: `beeja`

## Current stage

`bootstrap-complete` — next: `bigram`

## Milestones

| Stage | Status | Acceptance evidence |
|---|---|---|
| Bootstrap | Complete | `pyproject.toml` + `src/beeja` package installs editable; env report shows arm64 / 8 GiB / MPS available / torch 2.13.0; 12 tests pass; ruff clean |
| Bigram baseline | Not started | Loss decreases and text samples generate |
| Beeja-3M baseline | Not started | Approximately 3M parameters, causal attention tests, tiny-batch overfit, and checkpoint generation pass |
| Custom BPE tokenizer | Not started | Train/encode/decode and round-trip tests pass |
| Intermediate pretraining | Not started | Resumable TinyStories-style training pipeline |
| Modern architecture | Not started | RoPE, RMSNorm, SwiGLU comparisons pass |
| Instruction tuning | Not started | Assistant-only loss and chat formatting verified |
| Evaluation | Not started | Metrics and reproducible evaluation report |
| MLX/local inference | Not started | Local generation benchmark on Apple Silicon |
| Chat application | Not started | CLI or local web interface works |
| Final polish | Not started | Documentation, diagrams, results, demo |

## Last verified command

```bash
python3 -m venv .venv && .venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python scripts/check_environment.py   # arm64, 8 GiB, mps_available=true, torch 2.13.0
.venv/bin/python -m pytest                        # 12 passed
.venv/bin/ruff check . && .venv/bin/ruff format --check .   # clean
```

## Decisions

- Primary framework: PyTorch.
- Local accelerator: Apple MPS when available.
- Larger training: Google Colab with resumable checkpoints.
- Final parameter target: approximately 20–50M, adjusted using measured memory and speed.

## Open issues

- None yet.

## Next action

Run `/beeja-lab bigram`.
