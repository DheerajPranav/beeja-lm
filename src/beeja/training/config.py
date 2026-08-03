"""Training hyper-parameters for the pretraining pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    max_steps: int = 200
    warmup_steps: int = 20
    max_lr: float = 3e-3
    min_lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0  # 0 disables gradient clipping
    batch_size: int = 16
    block_size: int = 32  # window length; must not exceed model.block_size
    grad_accum_steps: int = 1  # effective batch = batch_size * grad_accum_steps
    eval_interval: int = 50  # 0 disables periodic validation
    eval_batches: int = 20
    checkpoint_interval: int = 100  # 0 disables periodic checkpoints
    sample_interval: int = 0  # 0 disables sample generation during training
    sample_tokens: int = 100
    seed: int = 1337
    device: str = "cpu"
    out_dir: str = "checkpoints"
    name: str = "Beeja-3M"
