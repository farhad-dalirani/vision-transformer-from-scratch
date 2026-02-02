from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:

    epochs: int = 300
    batch_size: int = 4096
    grad_clip_global_norm: float |None = 1.0
