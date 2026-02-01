import pytest
import torch
from torch.nn import CrossEntropyLoss

from vision_transformer.training.losses import (
    calculate_loss,
    get_criterion,
)


def test_wrong_criterion_name():

    with pytest.raises(ValueError):
        get_criterion(loss_name="wrong-criterion-name")


def test_get_cross_entropy_criterion():

    crt = get_criterion(loss_name="cross-entropy")
    assert isinstance(crt, CrossEntropyLoss)


def test_cross_entropy_set_hyperparameters():

    crt = get_criterion(loss_name="cross-entropy", label_smoothing=0.2, reduction="sum")
    assert crt.label_smoothing == 0.2
    assert crt.reduction == "sum"


def test_calculate_loss():
    crt = get_criterion(loss_name="cross-entropy")

    logit_pred = torch.rand(size=(4, 10))
    y_gt = torch.randint(low=0, high=10, size=(4,))

    loss = calculate_loss(
        criterion=crt,
        logit_pred=logit_pred,
        y_gt=y_gt,
    )

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
