#!/usr/bin/env python3
"""Build the Beeja-3M architecture, save an initial checkpoint, and write its model card.

This does NOT train the model — it materialises the first named release's
architecture from a fixed seed, measures the real parameter count, saves a
checkpoint under ``checkpoints/`` (gitignored), and regenerates the model card
under ``reports/``. Full pretraining comes in a later stage.

    python scripts/save_beeja3m.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from beeja.data.char import CharTokenizer  # noqa: E402
from beeja.data.sample import SAMPLE_TEXT  # noqa: E402
from beeja.models.config import beeja_3m_config  # noqa: E402
from beeja.models.gpt import BeejaGPT  # noqa: E402
from beeja.training.checkpoint import save_checkpoint  # noqa: E402
from beeja.utils import parameter_count, set_seed  # noqa: E402

NAME = "Beeja-3M"


def _model_card(config, counts: dict) -> str:
    return f"""# {NAME} — Model Card

**Status:** architecture materialised, **untrained** (random initialisation).
Full pretraining is a later stage; no quality metrics are claimed here.

## Identity
- Family: `Beeja`
- Release: `{NAME}`
- Type: decoder-only autoregressive Transformer (causal next-token prediction)

## Architecture
| field | value |
|---|---|
| vocab_size | {config.vocab_size} (character-level for this stage) |
| block_size (context) | {config.block_size} |
| n_layer | {config.n_layer} |
| n_head | {config.n_head} |
| n_embd | {config.n_embd} (head_size {config.head_size}) |
| dropout | {config.dropout} |
| normalization | pre-norm LayerNorm |
| activation | GELU (4x MLP) |

## Measured parameters
| component | count |
|---|---|
| embedding | {counts["embedding"]:,} |
| attention | {counts["attention"]:,} |
| mlp | {counts["mlp"]:,} |
| lm_head | {counts["lm_head"]:,} |
| norm/other | {counts["norm_other"]:,} |
| **total** | **{counts["total"]:,}** |

Parameter memory (fp32): {counts["param_mib"]} MiB. Training memory is larger:
add gradients (~1x), AdamW state (~2x), and activations (∝ batch × context × depth).

## Note on sizing
`architecture.md` lists d=128 as a starting point (~0.8M params). To make the
`Beeja-3M` name honest, `n_embd` was tuned to 256, giving ~3.2M measured params.
The character vocab is tiny, so embeddings are a negligible share; capacity lives
in the attention and MLP blocks.
"""


def main() -> int:
    set_seed(0)
    tokenizer = CharTokenizer.from_text(SAMPLE_TEXT)
    config = beeja_3m_config(vocab_size=tokenizer.vocab_size)
    model = BeejaGPT(config)
    counts = parameter_count(model)
    print(f"{NAME}: {counts['total']:,} parameters ({counts['param_mib']} MiB fp32)")

    ckpt_path = _ROOT / "checkpoints" / "beeja-3m-init.pt"
    save_checkpoint(ckpt_path, model, name=NAME, step=0, extra={"trained": False})
    print(f"saved checkpoint -> {ckpt_path.relative_to(_ROOT)}")

    card_path = _ROOT / "reports" / "beeja-3m-model-card.md"
    card_path.write_text(_model_card(config, counts))
    print(f"wrote model card -> {card_path.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
