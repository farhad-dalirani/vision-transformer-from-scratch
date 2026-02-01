from dataclasses import dataclass


@dataclass(frozen=True)
class LossConfig:
    """Configuration for loss construction."""

    loss_name: str = "cross-entropy"
    label_smoothing: float = 0.0
    reduction: str = "mean"
