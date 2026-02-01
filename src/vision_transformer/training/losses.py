from typing import Callable

import torch
from torch.nn import CrossEntropyLoss


def criterion(type="cross-entropy", **kwargs) -> Callable:
    """Factory function to create a loss criterion.

    Currently supports cross-entropy loss with optional label smoothing.

    Args:
        type: Type of loss to construct. Supported values:
            - "cross-entropy"
        **kwargs: Additional keyword arguments for the loss.
            label_smoothing: Amount of label smoothing to apply. Defaults to 0.0.
            reduction: Reduction method to apply to the loss. Defaults to "mean".

    Returns:
        A callable loss function that takes (logits, targets) and returns a scalar loss.

    Raises:
        ValueError: If an unsupported loss type is requested.
    """
    if type.lower() == "cross-entropy":
        label_smoothing = kwargs.get("label_smoothing", 0.0)
        reduction = kwargs.get("reduction", "mean")
        return cross_entropy_loss(label_smoothing=label_smoothing, reduction=reduction)
    else:
        raise ValueError(
            f"Requested criterion type {type} is not "
            "supported. Currently supported are ['cross-entropy']."
        )


def cross_entropy_loss(label_smoothing=0.0, reduction="mean") -> CrossEntropyLoss:
    """Constructs a cross-entropy loss function.

    This loss expects raw, unnormalized logits as input and integer class
    indices as targets.

    Args:
        label_smoothing: Amount of label smoothing to apply. A value of 0.0
            corresponds to standard cross-entropy loss without smoothing.
        reduction: Specifies the reduction to apply to the output:
            "none", "mean", or "sum".

    Returns:
        An instance of torch.nn.CrossEntropyLoss.
    """
    ce_loss = CrossEntropyLoss(label_smoothing=label_smoothing, reduction=reduction)
    return ce_loss


def calculate_loss(
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    logit_pred: torch.Tensor,
    y_gt: torch.Tensor,
) -> torch.Tensor:
    """Computes the classification loss from logits and ground-truth labels.

    This function applies the provided loss criterion directly to the raw
    logits produced by the model, as required for cross-entropy loss.

    Args:
        criterion: A loss function that takes logits and target labels.
        logit_pred: Model output logits of shape (batch_size, num_classes).
        y_gt: Ground-truth class indices of shape (batch_size,).

    Returns:
        A scalar tensor representing the computed loss.
    """
    loss = criterion(logit_pred, y_gt)
    return loss