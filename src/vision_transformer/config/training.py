from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    # Dataset / input
    dataset: str = "imagenet1k"
    num_classes: int = 1000
    image_size: int = 224

    # Training length
    epochs: int = 300

    # Batch / optimizer
    batch_size: int = 4096
    optimizer: str = "adam"
    betas: tuple = (0.9, 0.999)
    weight_decay: float = 0.3

    # Learning rate
    base_lr: float = 3e-3
    lr_schedule: str = "cosine"
    warmup_steps: int = 10_000

    # Regularization (ImageNet-from-scratch needs strong reg)
    dropout: float = 0.1
    label_smoothing: float = 0.0

    # Stability
    grad_clip_norm: float = 1.0
