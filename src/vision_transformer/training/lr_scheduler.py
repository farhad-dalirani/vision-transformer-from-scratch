from typing import Callable, Literal

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

SchedulerNames = Literal["cosine", "linear"]


def cosine_lr_scheduler_with_warmup_fn(
    total_steps: int,
    warmup_steps: int,
) -> Callable[[int], float]:
    """Creates a cosine learning-rate schedule with linear warm-up.

    The learning rate increases linearly from 0 to the base learning rate
    during the warm-up phase, then decays following a cosine curve to 0.

    Args:
        total_steps: Total number of optimizer steps during training.
        warmup_steps: Number of steps for linear warm-up.

    Returns:
        A callable that maps the current optimizer step to a multiplicative
        learning-rate factor.
    """

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return current_step / warmup_steps

        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + torch.cos(torch.pi * progress))

    return lr_lambda


def linear_lr_scheduler_with_warmup_fn(
    total_steps: int,
    warmup_steps: int,
) -> Callable[[int], float]:
    """Creates a linear learning-rate schedule with warm-up.

    The learning rate increases linearly from 0 to the base learning rate
    during warm-up, then decays linearly to 0 over the remaining steps.

    Args:
        total_steps: Total number of optimizer steps during training.
        warmup_steps: Number of steps for linear warm-up.

    Returns:
        A callable that maps the current optimizer step to a multiplicative
        learning-rate factor.
    """

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return current_step / warmup_steps

        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)

    return lr_lambda


def get_lr_scheduler(
    optimizer: Optimizer,
    scheduler_name: SchedulerNames,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """Creates a learning-rate scheduler with warm-up.

    Args:
        optimizer: Optimizer whose learning rate will be scheduled.
        scheduler_name: Type of scheduler to use ("cosine" or "linear").
        warmup_steps: Number of steps for linear warm-up.
        total_steps: Total number of optimizer steps during training.

    Returns:
        A PyTorch LambdaLR scheduler.

    Raises:
        ValueError: If the requested scheduler type is not supported.
    """

    scheduler_name = scheduler_name.lower()

    if scheduler_name == "cosine":
        lr_lambda = cosine_lr_scheduler_with_warmup_fn(
            total_steps=total_steps,
            warmup_steps=warmup_steps,
        )

    elif scheduler_name == "linear":
        lr_lambda = linear_lr_scheduler_with_warmup_fn(
            total_steps=total_steps,
            warmup_steps=warmup_steps,
        )

    else:
        raise ValueError(
            f"Requested scheduler type {scheduler_name} is not supported. "
            f"Supported schedulers: {SchedulerNames}"
        )

    return LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)
