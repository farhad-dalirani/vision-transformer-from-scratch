import pytest
import torch.nn as nn
from torch.optim import SGD, Adam

from vision_transformer.training.optim import get_optimizer


def test_wrong_optimizer_name():

    net = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3)

    with pytest.raises(ValueError):
        get_optimizer(net.parameters(), opt_name="wrong_name")


def test_get_adam_optimizer():
    net = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3)

    optimizer = get_optimizer(
        net.parameters(), opt_name="adam", lr=0.1, betas=(0.8, 0.9), weight_decay=0.0
    )

    assert isinstance(optimizer, Adam)

    group = optimizer.param_groups[0]
    assert group["lr"] == 0.1
    assert group["betas"] == (0.8, 0.9)
    assert group["weight_decay"] == 0.0


def test_get_sgd_optimizer():
    net = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3)

    optimizer = get_optimizer(
        net.parameters(), opt_name="sgd", lr=0.4, momentum=0.3, weight_decay=0.7
    )

    assert isinstance(optimizer, SGD)
    group = optimizer.param_groups[0]
    assert group["lr"] == 0.4
    assert group["momentum"] == 0.3
    assert group["weight_decay"] == 0.7
