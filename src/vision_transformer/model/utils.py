import torch.nn as nn


def get_activation_class_by_name(activation_function_name: str):
    """Returns a PyTorch activation module class by name.

    Args:
        activation_function_name: Name of the activation function
            ("relu", "gelu", or "tanh"), case-insensitive.

    Returns:
        A torch.nn.Module class corresponding to the activation function.

    Raises:
        ValueError: If the activation name is not supported.
    """
    function_name = activation_function_name.lower()
    if function_name == "relu":
        return nn.ReLU
    elif function_name == "gelu":
        return nn.GELU
    elif function_name == "tanh":
        return nn.Tanh
    else:
        raise ValueError(
            f"{function_name} is not supported, " 'options are ["relu", "gelu", "tanh"]'
        )
