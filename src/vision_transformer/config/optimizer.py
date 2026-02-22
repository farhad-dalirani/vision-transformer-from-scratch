from dataclasses import dataclass
from typing import Literal

optimizerName = Literal["adam", "sgd"]


@dataclass
class OptimizerConfig:
    """Configuration for optimizer construction.

    This dataclass defines the optimizer type and its associated
    hyperparameters. Some parameters are only relevant for specific
    optimizers.

    Attributes:
        opt_name (optimizerName): Name of the optimizer to use.
            Supported values:
                - "adam": Adam optimizer.
                - "sgd": Stochastic Gradient Descent optimizer.
            Defaults to "adam".

        betas (tuple[float, float]): Coefficients used for computing
            running averages of gradient and its square in Adam.
            Only used when `opt_name` is "adam".
            Defaults to (0.9, 0.999).

        momentum (float): Momentum factor for SGD.
            Only used when `opt_name` is "sgd".
            Defaults to 0.9.

        lr (float): Learning rate applied to parameter updates.
            Used by both Adam and SGD.
            Defaults to 8e-4.

        weight_decay (float): Weight decay (L2 regularization)
            coefficient. Used by both Adam and SGD.
            Defaults to 0.1.
    """

    opt_name: optimizerName = "adam"

    # Adam optimizer setting
    betas: tuple[float, float] = (0.9, 0.999)

    # SGD optimizer settings
    momentum: float = 0.9

    # Adam and SGD optimizer settings
    lr: float = 8e-4
    weight_decay: float = 0.1
