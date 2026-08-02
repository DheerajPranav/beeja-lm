# Beeja Project State

## Model identity

- Family: `Beeja`
- Current target: `Beeja-3M`
- Repository: `beeja-lm`
- Python package: `beeja`

## Current stage

`bigram-complete` — next: `transformer`

## Milestones

| Stage | Status | Acceptance evidence |
|---|---|---|
| Bootstrap | Complete | `pyproject.toml` + `src/beeja` package installs editable; env report shows arm64 / 8 GiB / MPS available / torch 2.13.0; 12 tests pass; ruff clean |
| Bigram baseline | Complete | Char tokenizer round-trips (ASCII/Unicode); batch shapes tested; loss 4.07→2.06 over 500 steps on embedded corpus; fixed-seed generation reproducible; 35 tests pass |
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

Bigram stage:

```bash
.venv/bin/python -m pytest                        # 35 passed
.venv/bin/python scripts/train_bigram.py --steps 500
  # vocab_size=38 chars=640 train=576 val=64; params=1444 (=38^2)
  # loss: first=4.0738 last=2.0630 val=2.1703; prints a fixed-seed sample
```

## Decisions

- Primary framework: PyTorch.
- Local accelerator: Apple MPS when available.
- Larger training: Google Colab with resumable checkpoints.
- Final parameter target: approximately 20–50M, adjusted using measured memory and speed.

## Open issues

- None yet.

## Next action

Run `/beeja-lab transformer`.
