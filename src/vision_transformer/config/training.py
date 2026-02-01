from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    # Dataset / input
    dataset: str = "cifar10"
    image_size: int = 32

    # Training length
    epochs: int = 300

    # Optimizer
    optimizer: str = "adam"
    betas: tuple[float, float] = (0.9, 0.999)
    batch_size: int = 4096
    weight_decay: float = 0.3

    # LR schedule
    base_lr: float = 3e-3
    lr_schedule: str = "cosine"
    warmup_steps: int = 10_000

    # Regularization
    dropout: float = 0.1

    # Loss function
    type: str = "cross-entropy"
    label_smoothing: float = 0.0
    reduction: str = "mean"

    # Stability
    grad_clip_norm: float = 1.0
