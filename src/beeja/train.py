"""Config-driven pretraining entry point.

    python -m beeja.train --config configs/smoke.yaml
    python -m beeja.train --config configs/smoke.yaml --resume checkpoints/Beeja-3M-step100.pt

The YAML has three sections: ``model``, ``data``, and ``training``. The data
pipeline (leakage-safe subword tokenization) and model vocab size are derived
from the corpus, so the config never has to hard-code the vocabulary.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from beeja.data.pipeline import build_datasets
from beeja.models.config import ModelConfig
from beeja.models.gpt import BeejaGPT
from beeja.training.config import TrainConfig
from beeja.training.trainer import Trainer
from beeja.utils import parameter_count, set_seed


def build_trainer(cfg: dict, *, device: str | None = None, resume: str | None = None) -> Trainer:
    data_cfg = cfg.get("data", {})
    model_cfg = cfg["model"]
    train_cfg = TrainConfig(**cfg.get("training", {}))
    if device:
        train_cfg.device = device

    train_ids, val_ids, tokenizer = build_datasets(
        source=data_cfg.get("source", "sample"),
        tokenizer_kind=data_cfg.get("tokenizer", "char"),
        vocab_size=data_cfg.get("vocab_size", 320),
        val_fraction=data_cfg.get("val_fraction", 0.1),
    )

    if resume:
        return Trainer.resume(resume, train_ids, val_ids, tokenizer, device=train_cfg.device)

    set_seed(train_cfg.seed)
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=model_cfg["block_size"],
        n_layer=model_cfg["n_layer"],
        n_head=model_cfg["n_head"],
        n_embd=model_cfg["n_embd"],
        dropout=model_cfg.get("dropout", 0.0),
    )
    # Keep the training window within the model's context length.
    train_cfg.block_size = model_config.block_size
    model = BeejaGPT(model_config)
    return Trainer(model, train_ids, val_ids, train_cfg, tokenizer)


def main() -> int:
    parser = argparse.ArgumentParser(description="Beeja pretraining")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    parser.add_argument("--device", default=None, help="override training device")
    parser.add_argument("--max-steps", type=int, default=None, help="override max_steps")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    trainer = build_trainer(cfg, device=args.device, resume=args.resume)
    if args.max_steps is not None:
        trainer.config.max_steps = args.max_steps

    counts = parameter_count(trainer.model)
    print(
        f"model: vocab={trainer.model.config.vocab_size} params={counts['total']:,} "
        f"| device={trainer.config.device} | start step={trainer.step}"
    )
    trainer.train(log=True)
    final = Path(trainer.config.out_dir) / f"{trainer.config.name}-final.pt"
    trainer.save(final)
    print(f"done at step {trainer.step}; saved -> {final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
