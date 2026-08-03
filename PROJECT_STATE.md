# Beeja Project State

## Model identity

- Family: `Beeja`
- Current target: `Beeja-3M`
- Repository: `beeja-lm`
- Python package: `beeja`

## Current stage

`pretrain-complete` — next: `modernize`

## Milestones

| Stage | Status | Acceptance evidence |
|---|---|---|
| Bootstrap | Complete | `pyproject.toml` + `src/beeja` package installs editable; env report shows arm64 / 8 GiB / MPS available / torch 2.13.0; 12 tests pass; ruff clean |
| Bigram baseline | Complete | Char tokenizer round-trips (ASCII/Unicode); batch shapes tested; loss 4.07→2.06 over 500 steps on embedded corpus; fixed-seed generation reproducible; 35 tests pass |
| Beeja-3M baseline | Complete | Measured 3,211,776 params (~3.21M, 12.25 MiB fp32); attention shapes + probs-sum-to-1 + no-future-leakage tests pass; end-to-end causal leakage test passes; tiny-batch overfit → loss <0.05; checkpoint round-trip preserves logits (atol 1e-6); `Beeja-3M` checkpoint + model card generated; 49 tests pass |
| Custom BPE tokenizer | Complete | Byte-level BPE from scratch; ASCII + Unicode (emoji/CJK) round trips; deterministic training; save/load identity; special-token handling; measured compression 2.46 bytes/token on held-out text; 61 tests pass |
| Intermediate pretraining | Complete | Config-driven resumable pipeline (AdamW + warmup/cosine + clip + accumulation); leakage-safe subword data; smoke config completes locally; exact resume verified (unit + CLI); grad-accum == full-batch; full-run config prepared, not launched; 75 tests pass |
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

Transformer (Beeja-3M) stage:

```bash
.venv/bin/python -m pytest                        # 49 passed
.venv/bin/python scripts/train_transformer.py --steps 500
  # smoke config (2L/d64), 107008 params; loss first=3.6670 last=0.2621 val=1.9760
  # sample now contains real words/phrases (attention uses long context)
.venv/bin/python scripts/save_beeja3m.py
  # Beeja-3M: 3,211,776 parameters (12.252 MiB fp32)
  # -> checkpoints/beeja-3m-init.pt (untrained, random init; gitignored)
  # -> reports/beeja-3m-model-card.md
```

Tokenizer (byte-level BPE) stage:

```bash
.venv/bin/python -m pytest                        # 61 passed
.venv/bin/python scripts/train_tokenizer.py --vocab-size 320
  # learned 64 merges; vocab_size=321 (256 bytes + 64 merges + 1 special)
  # held-out compression: 2.4615 bytes/token (byte-level baseline = 1.0)
  # -> checkpoints/beeja-bpe.json (gitignored)
```

Pretraining pipeline stage:

```bash
.venv/bin/python -m pytest                        # 75 passed
.venv/bin/python -m beeja.train --config configs/smoke.yaml
  # vocab=320 params=143,104; step200 loss=0.1988 val=5.1661 (tiny corpus overfits)
  # periodic eval + checkpoints + samples; -> checkpoints/Beeja-3M-final.pt
.venv/bin/python -m beeja.train --config configs/smoke.yaml \
    --resume checkpoints/Beeja-3M-step100.pt
  # resumes at step 100 -> reproduces step200 loss=0.1988 val=5.1661 exactly
```

Full run (PREPARED, NOT launched):

```bash
python scripts/download_data.py                          # fetch corpus into data/
python -m beeja.train --config configs/beeja-10m.yaml --device cuda   # Colab/GPU
```

## Decisions

- Primary framework: PyTorch.
- Local accelerator: Apple MPS when available.
- Larger training: Google Colab with resumable checkpoints.
- Final parameter target: approximately 20–50M, adjusted using measured memory and speed.

## Open issues

- None yet.

## Next action

Run `/beeja-lab modernize`.
