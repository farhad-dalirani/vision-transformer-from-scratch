import pytest
import torch.nn as nn

from vision_transformer.model.utils import get_activation_class_by_name


def test_return_activation_class():

    assert get_activation_class_by_name(activation_function_name="relu") is nn.ReLU
    assert get_activation_class_by_name(activation_function_name="GELU") is nn.GELU
    assert get_activation_class_by_name(activation_function_name="TaNh") is nn.Tanh


def test_activation_name_should_be_supported():
    with pytest.raises(ValueError):
        get_activation_class_by_name(activation_function_name="wrong_name")
