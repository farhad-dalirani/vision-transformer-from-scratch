import pytest
import torch

from vision_transformer.model.transformer_encoder_block import TransformerEncoderBlock


def test_transformer_encoder_block_output_shape():

    encoder_block = TransformerEncoderBlock(
        embed_dim=100,
        num_heads=10,
        mha_attention_dropout_p=0.1,
        mha_proj_dropout_p=0.1,
        mlp_ratio=4.0,
        mlp_dropout_p=0.1,
    )

    x = torch.rand(size=(3, 5, 100))
    out = encoder_block(x)

    assert out.shape == (3, 5, 100)


def test_transformer_encoder_block_backward_pass():

    encoder_block = TransformerEncoderBlock(
        embed_dim=100,
        num_heads=10,
        mha_attention_dropout_p=0.1,
        mha_proj_dropout_p=0.1,
        mlp_ratio=4.0,
        mlp_dropout_p=0.1,
    )

    x = torch.rand(size=(3, 5, 100), requires_grad=True)
    out = encoder_block(x)
    score = out.sum()
    score.backward()

    assert x.grad is not None
    assert encoder_block.norm1.weight.grad is not None
    assert encoder_block.norm2.weight.grad is not None
