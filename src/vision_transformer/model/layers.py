import torch
import torch.nn as nn

from vision_transformer.model.utils import get_activation_class_by_name


class MLPBlock(nn.Module):
    """Feed-forward MLP block: stacked Linear layers with an activation after each
    intermediate layer and per-layer dropout.

    Args:
        input_dim: Input feature dimension.
        layers_dim: Output dimension for each Linear layer (length = num_layers).
        layers_dropout_p: Dropout probability applied after each Linear layer;
            must have the same length as `layers_dim`.
        activation_type: Activation name for intermediate layers (e.g., "gelu").

    Shape:
        Input:  (..., input_dim)
        Output: (..., layers_dim[-1])
    """

    def __init__(
        self,
        input_dim: int,
        layers_dim: list[int],
        layers_dropout_p: list[float],
        activation_type: str,
    ):
        super().__init__()

        if len(layers_dropout_p) != len(layers_dim):
            raise ValueError("layers_dropout_p must have same length as layers_dim")

        self.activation_function = get_activation_class_by_name(
            activation_function_name=activation_type
        )()

        self.dropouts = nn.ModuleList([nn.Dropout(p=p_i) for p_i in layers_dropout_p])

        self.layers = nn.ModuleList()
        for i in range(len(layers_dim)):
            if i == 0:
                in_features = input_dim
            else:
                in_features = layers_dim[i - 1]
            self.layers.append(
                nn.Linear(in_features=in_features, out_features=layers_dim[i])
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_layers = len(self.layers)
        out = x
        for i in range(num_layers):
            out = self.layers[i](out)
            if i != num_layers - 1:
                out = self.activation_function(out)
            out = self.dropouts[i](out)
        return out
