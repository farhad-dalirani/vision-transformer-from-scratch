from dataclasses import dataclass
from typing import Literal

SchedulerNames = Literal["cosine", "linear", "none"]


@dataclass(frozen=True)
class LRSchedulerConfig:
    """Configuration for a learning rate scheduler.

    This dataclass defines the scheduler type and warmup behavior used
    during training.

    Attributes:
        scheduler_name (SchedulerNames): Name of the learning rate scheduler.
            Supported values are:
                - "cosine": Cosine decay schedule.
                - "linear": Linear decay schedule.
                - "none": No scheduling applied.
            Defaults to "cosine".
        warmup_steps (int): Number of warmup steps during which the learning
            rate increases linearly from zero to the base learning rate.
            Defaults to 10_000.
    """

    scheduler_name: SchedulerNames = "cosine"
    warmup_steps: int = 10_000
