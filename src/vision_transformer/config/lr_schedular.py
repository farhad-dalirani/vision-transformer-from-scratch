from dataclasses import dataclass


@dataclass(frozen=True)
class LRSchedulerConfig:

    scheduler_name: str = "cosine"
    warmup_steps: int = 10_000
