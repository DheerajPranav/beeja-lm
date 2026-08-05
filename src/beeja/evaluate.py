"""Evaluate a trained checkpoint and write a reproducible report.

    python -m beeja.evaluate --config configs/beeja-3m-tinystories.yaml \
        --checkpoint checkpoints/Beeja-3M-TinyStories-final.pt

Rebuilds the tokenizer and validation set from the config, rebuilds the model
from the config stored in the checkpoint, computes metrics + a runtime benchmark,
and writes a Markdown report under reports/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from beeja.data.pipeline import build_datasets
from beeja.evaluation.report import evaluation_report, render_markdown
from beeja.models.config import ModelConfig
from beeja.models.gpt import BeejaGPT
from beeja.utils import set_seed


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a Beeja checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--out", default=None, help="report path (default reports/eval-<name>.md)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg.get("data", {})
    seed = cfg.get("training", {}).get("seed", 1337)
    set_seed(seed)

    _, val_ids, tokenizer = build_datasets(
        source=data_cfg.get("source", "sample"),
        tokenizer_kind=data_cfg.get("tokenizer", "char"),
        vocab_size=data_cfg.get("vocab_size", 320),
        val_fraction=data_cfg.get("val_fraction", 0.1),
        max_chars=data_cfg.get("max_chars"),
        tokenizer_max_chars=data_cfg.get("tokenizer_max_chars"),
    )

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model_config = ModelConfig(**ckpt["model_config"])
    model = BeejaGPT(model_config)
    model.load_state_dict(ckpt["model_state"])
    model.to(args.device).eval()
    name = ckpt.get("name", "Beeja")

    report = evaluation_report(
        model,
        val_ids,
        tokenizer,
        model_config=model_config,
        block_size=model_config.block_size,
        eval_batches=args.eval_batches,
        seed=seed,
        device=args.device,
    )

    out_path = Path(args.out) if args.out else Path("reports") / f"eval-{name}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_markdown(report, name=name))
    (out_path.with_suffix(".json")).write_text(json.dumps(report, indent=2, default=str))

    print(
        f"perplexity={report['metrics']['perplexity']:.3f} "
        f"bits/token={report['metrics']['bits_per_token']:.4f} "
        f"tok/s={report['benchmark']['tokens_per_second']}"
    )
    print(f"report -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
