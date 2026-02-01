from dataclasses import dataclass
from typing import Literal

optimizerName = Literal["adam", "sgd"]


@dataclass
class OptimizerConfig:
    type: optimizerName = "adam"

    # Adam optimizer setting
    betas: tuple[float, float] = (0.9, 0.999)

    # SGD optimizer settings
    momentum: float = 0.9

    # Adam and SGD optimizer settings
    lr: float = 8e-4
    weight_decay: float = 0.1
