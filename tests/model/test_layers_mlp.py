import pytest
import torch

from vision_transformer.model.mlp_block import (
    FinetuningClassificationHead,
    MLPBlock,
    PreTrainingClassificationHead,
    TransformerMLP,
)


def test_input_output_dims():
    x = torch.ones(size=(5, 21))

    mlp_block = MLPBlock(
        input_dim=21,
        layers_dim=[84, 10],
        layers_dropout_p=[0.1, 0.2],
        activation_type="gelu",
    )

    out = mlp_block(x)
    assert out.shape == (5, 10)


def test_dropout_list_length_must_match_layers_dim():
    with pytest.raises(ValueError):
        MLPBlock(
            input_dim=21,
            layers_dim=[84, 10],
            layers_dropout_p=[0.1],  # wrong length
            activation_type="gelu",
        )


def test_eval_mode_is_deterministic():
    torch.manual_seed(0)
    x = torch.randn(5, 21)

    mlp_block = MLPBlock(
        input_dim=21,
        layers_dim=[84, 10],
        layers_dropout_p=[0.5, 0.5],
        activation_type="gelu",
    ).eval()

    out1 = mlp_block(x)
    out2 = mlp_block(x)

    assert torch.allclose(out1, out2)


def test_transformer_mlp_shape():

    mlp = TransformerMLP(embed_dim=100, mlp_ratio=4.0, dropout_p=0.2)

    input_tensor = torch.rand(size=(4, 100))
    out = mlp(input_tensor)

    assert out.shape == (4, 100)


def test_transformer_mlp_hidden_dim_is_Nx_embed_dim():
    embed_dim = 100
    mlp_ratio = 4.0

    expanded_layer_dim = int(mlp_ratio * embed_dim)

    mlp = TransformerMLP(embed_dim=embed_dim, mlp_ratio=mlp_ratio, dropout_p=0.2)

    linear = mlp.layers[0]
    assert linear.in_features == embed_dim
    assert linear.out_features == expanded_layer_dim

    linear = mlp.layers[1]
    assert linear.in_features == expanded_layer_dim
    assert linear.out_features == embed_dim


def test_transformer_pretraining_mlp_head_layers_dim():

    input_dim = 100
    num_classes = 17

    mlp = PreTrainingClassificationHead(input_dim=input_dim, num_classes=num_classes)

    linear = mlp.layers[0]
    assert linear.in_features == input_dim
    assert linear.out_features == input_dim

    linear = mlp.layers[1]
    assert linear.in_features == input_dim
    assert linear.out_features == num_classes


def test_transformer_pretraining_mlp_head_output_dim():

    input_dim = 100
    num_classes = 17

    mlp = PreTrainingClassificationHead(input_dim=input_dim, num_classes=num_classes)

    in_tensor = torch.rand(size=(5, 100))
    out = mlp(in_tensor)

    assert out.shape == (5, num_classes)


def test_transformer_finetuning_mlp_head_output_dim():

    input_dim = 100
    num_classes = 17

    mlp = FinetuningClassificationHead(input_dim=input_dim, num_classes=num_classes)

    in_tensor = torch.rand(size=(5, 100))
    out = mlp(in_tensor)

    assert out.shape == (5, num_classes)
