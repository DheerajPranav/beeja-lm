#!/usr/bin/env python3
"""Baseline vs modern architecture under an identical controlled smoke setup.

Trains two same-sized models on the same data, seed, and schedule:
  - baseline: learned positions + LayerNorm + GELU MLP, untied head
  - modern:   RoPE + RMSNorm + SwiGLU, weight-tied head
Reports measured parameter counts, parameter memory, and final train/val loss.

This is a controlled comparison of the *machinery* on a tiny corpus, not a claim
about which architecture is better at scale.

    python scripts/compare_architectures.py --steps 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from beeja.data.pipeline import build_datasets  # noqa: E402
from beeja.models.config import ModelConfig  # noqa: E402
from beeja.models.gpt import BeejaGPT  # noqa: E402
from beeja.training.config import TrainConfig  # noqa: E402
from beeja.training.trainer import Trainer  # noqa: E402
from beeja.utils import parameter_count, set_seed  # noqa: E402

BLOCK = 16
DIMS = dict(block_size=BLOCK, n_layer=2, n_head=4, n_embd=64)


def run(name: str, config: ModelConfig, train_ids, val_ids, steps: int) -> dict:
    set_seed(0)
    model = BeejaGPT(config)
    counts = parameter_count(model)
    tcfg = TrainConfig(
        max_steps=steps,
        warmup_steps=max(1, steps // 10),
        batch_size=16,
        block_size=BLOCK,
        eval_interval=0,
        checkpoint_interval=0,
        sample_interval=0,
        seed=1337,
        device="cpu",
    )
    trainer = Trainer(model, train_ids, val_ids, tcfg)
    history = trainer.train()
    return {
        "name": name,
        "params": counts["total"],
        "mib": counts["param_mib"],
        "first": history[0],
        "last": history[-1],
        "val": trainer.evaluate(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()

    train_ids, val_ids, tok = build_datasets("sample", "bpe", vocab_size=320, val_fraction=0.1)
    v = tok.vocab_size

    baseline = ModelConfig(vocab_size=v, **DIMS)
    modern = ModelConfig(
        vocab_size=v, **DIMS, pos_encoding="rope", norm="rmsnorm", mlp="swiglu", tie_weights=True
    )

    rows = [
        run("baseline", baseline, train_ids, val_ids, args.steps),
        run("modern", modern, train_ids, val_ids, args.steps),
    ]

    print(f"\n{'arch':10} {'params':>9} {'MiB':>7} {'loss0':>8} {'lossN':>8} {'val':>8}")
    print("-" * 54)
    for r in rows:
        print(
            f"{r['name']:10} {r['params']:>9,} {r['mib']:>7.2f} "
            f"{r['first']:>8.4f} {r['last']:>8.4f} {r['val']:>8.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
