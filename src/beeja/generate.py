"""Generate text from a trained checkpoint.

    python -m beeja.generate --config configs/beeja-3m-tinystories.yaml \
        --checkpoint checkpoints/Beeja-3M-TinyStories-final.pt \
        --prompt "Once upon a time" --max-new-tokens 200

The tokenizer is rebuilt from the config's data section (deterministic), and the
model is rebuilt from the config stored inside the checkpoint.
"""

from __future__ import annotations

import argparse

import torch
import yaml

from beeja.data.pipeline import build_datasets
from beeja.models.config import ModelConfig
from beeja.models.gpt import BeejaGPT
from beeja.utils import set_seed


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate text from a Beeja checkpoint")
    parser.add_argument(
        "--config", required=True, help="config used for training (for the tokenizer)"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    set_seed(args.seed)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg.get("data", {})

    # Rebuild the tokenizer exactly as training did.
    _, _, tokenizer = build_datasets(
        source=data_cfg.get("source", "sample"),
        tokenizer_kind=data_cfg.get("tokenizer", "char"),
        vocab_size=data_cfg.get("vocab_size", 320),
        val_fraction=data_cfg.get("val_fraction", 0.1),
        max_chars=data_cfg.get("max_chars"),
        tokenizer_max_chars=data_cfg.get("tokenizer_max_chars"),
    )

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = BeejaGPT(ModelConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model_state"])
    model.to(args.device).eval()

    ids = tokenizer.encode(args.prompt) if args.prompt else [0]
    idx = torch.tensor([ids], dtype=torch.long, device=args.device)
    out = model.generate(idx, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
    print(tokenizer.decode(out[0].tolist()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
