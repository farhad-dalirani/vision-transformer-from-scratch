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


class TransformerMLP(MLPBlock):
    """Feed-forward MLP used inside a Transformer encoder block.

    Implements the position-wise feed-forward network (FFN) described
    in the Vision Transformer (ViT) paper. The MLP consists of two linear
    layers with a GELU activation and dropout applied after each dense
    layer.

    Architecture:
        Linear(D -> mlp_ratio * D) -> GELU -> Dropout
        -> Linear(mlp_ratio * D -> D) -> Dropout

    Args:
        embed_dim: Token embedding dimension (D).
        mlp_ratio: Expansion ratio for the hidden layer. For example, with
            mlp_ratio=4.0 and embed_dim=10, the hidden layer has 40 units.
        dropout_p: Dropout probability applied after each dense layer.
    """

    def __init__(
        self,
        embed_dim: int,
        mlp_ratio: float,
        dropout_p: float,
    ):
        hidden_dim = int(embed_dim * mlp_ratio)

        super().__init__(
            input_dim=embed_dim,
            layers_dim=[hidden_dim, embed_dim],
            layers_dropout_p=[dropout_p, dropout_p],
            activation_type="gelu",
        )


class PreTrainingClassificationHead(MLPBlock):
    """Pre-training classification head for Vision Transformers.

    Implements the MLP head used during supervised pre-training in the
    ViT paper: a single hidden layer with tanh activation followed by
    a linear classifier. This head is applied to the [CLS] token output
    of the Transformer encoder and is replaced during fine-tuning.

    Architecture:
        Linear(D -> D) -> tanh -> Linear(D → K)

    Args:
        input_dim: Embedding dimension of the [CLS] token.
        num_classes: Number of classes in the pre-training dataset.
    """

    def __init__(self, input_dim: int, num_classes: int):
        activation_type = "tanh"
        layers_dropout_p = [0.0, 0.0]
        layers_dim = [input_dim, num_classes]
        super().__init__(input_dim, layers_dim, layers_dropout_p, activation_type)


class FinetuningClassificationHead(nn.Module):
    """Fine-tuning classification head for Vision Transformers.

    Implements the linear classification head used during fine-tuning,
    as described in the ViT paper. This head replaces the pre-training
    MLP head and consists of a single linear projection applied to the
    [CLS] token.

    Architecture:
        Linear(D → K)

    Args:
        input_dim: Embedding dimension of the [CLS] token.
        num_classes: Number of target classes for fine-tuning.
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
