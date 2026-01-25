import pytest
import torch

from vision_transformer.model.attention import MultiHeadSelfAttention


def test_multi_head_attention_shape():

    mha = MultiHeadSelfAttention(
        dim=125, num_heads=5, attention_dropout_p=0.1, proj_dropout_p=0.1
    )

    in_tensor = torch.rand(size=(5, 20, 125))  # B, N, D
    out = mha(in_tensor)

    assert out.shape == (5, 20, 125)


def test_multi_head_attention_token_dim_divisible_by_num_head():

    with pytest.raises(ValueError):
        _ = MultiHeadSelfAttention(
            dim=125, num_heads=7, attention_dropout_p=0.1, proj_dropout_p=0.1
        )
