#!/usr/bin/env python3
"""Smoke-train the decoder-only Transformer on the embedded sample text.

Uses the small ``smoke`` config so it runs in seconds on CPU. This is a pipeline
check (loss goes down, generation runs), NOT a real training run. Prints the
train/val loss and a fixed-seed sample.

    python scripts/train_transformer.py --steps 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402

from beeja.data.char import CharTokenizer  # noqa: E402
from beeja.data.dataset import encode_dataset, train_val_split  # noqa: E402
from beeja.data.sample import SAMPLE_TEXT  # noqa: E402
from beeja.models.config import smoke_config  # noqa: E402
from beeja.models.gpt import BeejaGPT  # noqa: E402
from beeja.training.basic import evaluate_loss, fit  # noqa: E402
from beeja.utils import parameter_count, set_seed  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    args = parser.parse_args()

    set_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    tokenizer = CharTokenizer.from_text(SAMPLE_TEXT)
    data = encode_dataset(SAMPLE_TEXT, tokenizer)
    train_data, val_data = train_val_split(data, val_fraction=0.1)

    config = smoke_config(tokenizer.vocab_size)
    model = BeejaGPT(config)
    print(f"config: {config}")
    print(f"parameters: {parameter_count(model)['total']}")

    losses = fit(
        model,
        train_data,
        block_size=config.block_size,
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        generator=generator,
    )
    val_loss = evaluate_loss(
        model,
        val_data,
        block_size=config.block_size,
        batch_size=args.batch_size,
        batches=20,
        generator=generator,
    )
    print(f"loss: first={losses[0]:.4f}  last={losses[-1]:.4f}  val={val_loss:.4f}")

    start = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(start, args.max_new_tokens, generator=generator)
    print("--- sample ---")
    print(tokenizer.decode(out[0].tolist()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
