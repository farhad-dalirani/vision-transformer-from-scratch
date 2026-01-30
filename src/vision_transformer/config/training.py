from dataclasses import dataclass

@dataclass(frozen=True)
class TrainingConfig:
    # Dataset / input
    dataset: str = "imagenet1k"
    num_classes: int = 1000
    image_size: int = 224

    # Training length (ImageNet)
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

    # Stability
    grad_clip_norm: float = 1.0
