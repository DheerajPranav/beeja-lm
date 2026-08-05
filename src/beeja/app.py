"""Interactive streaming completion CLI for a trained Beeja checkpoint.

    python -m beeja.app --config configs/beeja-3m-tinystories.yaml \
        --checkpoint checkpoints/Beeja-3M-TinyStories-final.pt

Type a prompt and watch the model stream a continuation (KV-cached). This is a
*base* model, so it continues text rather than following instructions. One-shot
mode: pass --prompt "..." to generate once and exit.

Empty line or Ctrl-D/Ctrl-C exits.
"""

from __future__ import annotations

import argparse
import sys

from beeja.inference import build_tokenizer, load_config, load_model, stream_completion
from beeja.utils import set_seed


def _complete(model, tokenizer, prompt, args) -> None:
    print(prompt, end="", flush=True)
    for delta in stream_completion(
        model,
        tokenizer,
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
    ):
        print(delta, end="", flush=True)
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Beeja interactive completion CLI")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prompt", default=None, help="one-shot prompt; omit for interactive REPL")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    set_seed(args.seed)
    try:
        cfg = load_config(args.config)
        tokenizer = build_tokenizer(cfg)
        model, ckpt = load_model(args.checkpoint, device=args.device)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.prompt is not None:
        _complete(model, tokenizer, args.prompt, args)
        return 0

    print(f"Beeja ({ckpt.get('name', 'model')}) — type a prompt, empty line to quit.\n")
    while True:
        try:
            prompt = input("beeja> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt.strip():
            break
        _complete(model, tokenizer, prompt, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
