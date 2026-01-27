import pytest
import torch

from vision_transformer.model.embeddings import CLSTokenAndPositionEmbedding


def test_cls_token_and_positional_embedding_plus_one_token():

    cls_pos_embedding = CLSTokenAndPositionEmbedding(
        num_patches=100, embed_dim=600, dropout_p=0.0
    )

    in_tensor = torch.rand(size=(14, 100, 600))
    out = cls_pos_embedding(in_tensor)

    assert out.shape == (14, 101, 600)


def test_num_patches_must_be_perfect_square():

    with pytest.raises(ValueError):
        CLSTokenAndPositionEmbedding._grid_size(num_patches=10)


def test_grid_size():
    assert CLSTokenAndPositionEmbedding._grid_size(num_patches=16) == (4, 4)


def test_size_of_positional_embedding():

    cls_pos_embedding = CLSTokenAndPositionEmbedding(
        num_patches=100, embed_dim=600, dropout_p=0.0
    )
    assert cls_pos_embedding.pos_embed.shape == (1, 100 + 1, 600)


def test_shape_of_cls_token():
    cls_pos_embedding = CLSTokenAndPositionEmbedding(
        num_patches=100, embed_dim=600, dropout_p=0.0
    )
    assert cls_pos_embedding.cls_token.shape == (1, 1, 600)


def test_check_num_positional_embed_after_interpolation():

    cls_pos_embedding = CLSTokenAndPositionEmbedding(
        num_patches=100, embed_dim=600, dropout_p=0.0
    )

    in_tensor = torch.rand(size=(14, 100, 600))
    out = cls_pos_embedding(in_tensor)
    assert out.shape == (14, 101, 600)

    in_tensor = torch.rand(size=(14, 15 * 15, 600))
    out = cls_pos_embedding(in_tensor)
    assert out.shape == (14, 15 * 15 + 1, 600)


def test_cls_token_is_prepended_and_position_added():
    m = CLSTokenAndPositionEmbedding(num_patches=16, embed_dim=8, dropout_p=0.0)
    x = torch.zeros(2, 16, 8)

    y = m(x)  # (2, 17, 8)

    expected = m.cls_token.expand(2, -1, -1) + m.pos_embed[:, :1, :]
    assert torch.allclose(y[:, :1, :], expected)


def test_patch_tokens_equal_pos_embed_when_input_zeros():
    m = CLSTokenAndPositionEmbedding(num_patches=16, embed_dim=8, dropout_p=0.0)
    x = torch.zeros(1, 16, 8)

    y = m(x)
    assert torch.allclose(y[:, 1:, :], m.pos_embed[:, 1:, :])


def test_interpolation_keeps_cls_position_embedding():
    m = CLSTokenAndPositionEmbedding(num_patches=16, embed_dim=8, dropout_p=0.0)
    pos_new = m.interpolate_positional_embeddings(new_num_patches=25)  # 5x5

    assert torch.allclose(pos_new[:, :1, :], m.pos_embed[:, :1, :])


def test_backward_pass():
    m = CLSTokenAndPositionEmbedding(num_patches=16, embed_dim=8, dropout_p=0.0)
    x = torch.randn(2, 16, 8, requires_grad=True)

    y = m(x).sum()
    y.backward()

    assert x.grad is not None
    assert m.cls_token.grad is not None
    assert m.pos_embed.grad is not None


def test_raises_on_embed_dim_mismatch():
    m = CLSTokenAndPositionEmbedding(num_patches=16, embed_dim=8, dropout_p=0.0)
    x = torch.randn(2, 16, 7)
    with pytest.raises(ValueError):
        _ = m(x)
