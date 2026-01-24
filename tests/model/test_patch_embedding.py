import pytest
import torch

from vision_transformer.model.patch_embedding import PatchEmbedding


def test_patch_embedding_output_shape():

    patch_embed_layer = PatchEmbedding(
        image_size=224, in_channels=3, patch_size=16, embed_dim=768, strict=True
    )
    input_tensor = torch.rand(size=(5, 3, 224, 224))
    out = patch_embed_layer(input_tensor)
    assert out.shape == (5, 196, 768)  # B, N, D

    patch_embed_layer = PatchEmbedding(
        image_size=224, in_channels=1, patch_size=16, embed_dim=768, strict=True
    )
    input_tensor = torch.rand(size=(5, 1, 224, 224))
    out = patch_embed_layer(input_tensor)
    assert out.shape == (5, 196, 768)  # B, N, D


def test_patch_embedding_raise_error_different_res_strict_mode_true():

    patch_embed_layer = PatchEmbedding(
        image_size=224, in_channels=3, patch_size=16, embed_dim=768, strict=True
    )
    input_tensor = torch.rand(size=(5, 3, 448, 448))
    with pytest.raises(ValueError):
        _ = patch_embed_layer(input_tensor)


def test_patch_embedding_different_res_strict_mode_false():

    patch_embed_layer = PatchEmbedding(
        image_size=224, in_channels=3, patch_size=16, embed_dim=768, strict=False
    )
    input_tensor = torch.rand(size=(5, 3, 448, 448))
    out = patch_embed_layer(input_tensor)
    assert out.shape == (5, 784, 768)  # B, N, D


def test_patch_embedding_raises_when_input_not_divisible_by_patch_size():

    patch_embed_layer = PatchEmbedding(
        image_size=224, in_channels=3, patch_size=16, embed_dim=768, strict=True
    )
    input_tensor = torch.rand(size=(5, 3, 777, 448))
    with pytest.raises(ValueError):
        _ = patch_embed_layer(input_tensor)


def test_patch_embedding_raises_on_channel_mismatch_in_strict_mode():
    layer = PatchEmbedding(
        image_size=224, in_channels=3, patch_size=16, embed_dim=768, strict=True
    )
    x = torch.rand(5, 1, 224, 224)  # wrong channels
    with pytest.raises(ValueError):
        _ = layer(x)
