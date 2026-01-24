import pytest
import torch

from vision_transformer.model.layers import MLPBlock


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
