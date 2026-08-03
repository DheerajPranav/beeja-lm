"""Resumable pretraining loop.

Puts together the pieces required for real (if small) pretraining:

- AdamW with weight decay and betas (0.9, 0.95).
- Linear-warmup + cosine-decay learning rate (``schedule.lr_at``).
- Gradient accumulation: effective batch = ``batch_size * grad_accum_steps``.
- Gradient clipping by global norm.
- Periodic validation (no grad, no state update) and sample generation.
- Full-state checkpointing (model, optimizer, step, RNG) and exact resume.

Resume correctness relies on saving the batch-sampling generator's state along
with the optimizer state, so an interrupted run continues on the same trajectory
as an uninterrupted one.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from beeja.data.dataset import get_batch
from beeja.models.config import ModelConfig
from beeja.models.gpt import BeejaGPT
from beeja.training.basic import evaluate_loss
from beeja.training.config import TrainConfig
from beeja.training.schedule import lr_at

FORMAT_VERSION = 1


class Trainer:
    def __init__(
        self,
        model: BeejaGPT,
        train_data: torch.Tensor,
        val_data: torch.Tensor,
        config: TrainConfig,
        tokenizer: Any | None = None,
    ) -> None:
        if config.block_size > model.config.block_size:
            raise ValueError(
                f"train block_size {config.block_size} exceeds model block_size "
                f"{model.config.block_size}"
            )
        self.model = model.to(config.device)
        self.train_data = train_data
        self.val_data = val_data
        self.config = config
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.max_lr,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
        )
        # Dedicated generator for batch sampling; its state is checkpointed so
        # resume reproduces the exact batch sequence.
        self.generator = torch.Generator().manual_seed(config.seed)
        self.step = 0
        self.best_val = float("inf")

    def _set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def train_step(self) -> tuple[float, float]:
        """Run one optimizer step (with accumulation). Returns (loss, lr)."""
        cfg = self.config
        lr = lr_at(
            self.step,
            warmup_steps=cfg.warmup_steps,
            max_steps=cfg.max_steps,
            max_lr=cfg.max_lr,
            min_lr=cfg.min_lr,
        )
        self._set_lr(lr)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        for _ in range(cfg.grad_accum_steps):
            x, y = get_batch(
                self.train_data,
                block_size=cfg.block_size,
                batch_size=cfg.batch_size,
                generator=self.generator,
                device=cfg.device,
            )
            _, loss = self.model(x, y)
            # Divide so summed micro-grads equal the full-batch mean gradient.
            (loss / cfg.grad_accum_steps).backward()
            total_loss += loss.item() / cfg.grad_accum_steps

        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
        self.optimizer.step()
        self.step += 1
        return total_loss, lr

    @torch.no_grad()
    def evaluate(self) -> float:
        """Validation loss. Uses a fixed generator so the metric is stable, and
        never touches training RNG, optimizer, or parameters."""
        return evaluate_loss(
            self.model,
            self.val_data,
            block_size=self.config.block_size,
            batch_size=self.config.batch_size,
            batches=self.config.eval_batches,
            device=self.config.device,
            generator=torch.Generator().manual_seed(self.config.seed),
        )

    def train(self, until_step: int | None = None, log: bool = False) -> list[float]:
        cfg = self.config
        target = cfg.max_steps if until_step is None else until_step
        history: list[float] = []
        while self.step < target:
            loss, lr = self.train_step()
            if not math.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {self.step}: {loss}")
            history.append(loss)

            if cfg.eval_interval and self.step % cfg.eval_interval == 0:
                val = self.evaluate()
                self.best_val = min(self.best_val, val)
                if log:
                    print(f"step {self.step:5d} | loss {loss:.4f} | val {val:.4f} | lr {lr:.2e}")
            if cfg.checkpoint_interval and self.step % cfg.checkpoint_interval == 0:
                self.save(Path(cfg.out_dir) / f"{cfg.name}-step{self.step}.pt")
            if cfg.sample_interval and self.tokenizer and self.step % cfg.sample_interval == 0:
                if log:
                    print("  sample:", self._sample())
        return history

    @torch.no_grad()
    def _sample(self) -> str:
        start = torch.zeros((1, 1), dtype=torch.long, device=self.config.device)
        out = self.model.generate(
            start, self.config.sample_tokens, generator=torch.Generator().manual_seed(0)
        )
        return self.tokenizer.decode(out[0].tolist())

    # -- checkpointing ------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "name": self.config.name,
            "step": self.step,
            "best_val": self.best_val,
            "model_config": asdict(self.model.config),
            "train_config": asdict(self.config),
            "model_state": self.model.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "generator_state": self.generator.get_state(),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self.model.load_state_dict(sd["model_state"])
        self.optimizer.load_state_dict(sd["optim_state"])
        self.generator.set_state(sd["generator_state"])
        self.step = sd["step"]
        self.best_val = sd["best_val"]

    @classmethod
    def resume(
        cls,
        path: str | Path,
        train_data: torch.Tensor,
        val_data: torch.Tensor,
        tokenizer: Any | None = None,
        device: str | None = None,
    ) -> Trainer:
        ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
        model_config = ModelConfig(**ckpt["model_config"])
        train_config = TrainConfig(**ckpt["train_config"])
        if device:
            train_config.device = device
        model = BeejaGPT(model_config)
        trainer = cls(model, train_data, val_data, train_config, tokenizer)
        trainer.load_state_dict(ckpt)
        return trainer
