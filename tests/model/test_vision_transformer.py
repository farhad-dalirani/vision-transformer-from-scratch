import pytest
import torch

from vision_transformer.model.vision_transformer import VisionTransformer


def _make_vit(head_type="pretrain", strict=True, encoder_final_norm=True):
    return VisionTransformer(
        image_size=144,
        in_channels=3,
        patch_size=12,
        embed_dim=128,
        strict=strict,
        positional_embedding_dropout_p=0.1,
        num_attention_heads=8,
        mha_attention_dropout_p=0.1,
        mha_proj_dropout_p=0.1,
        mlp_ratio=4.0,
        mlp_dropout_p=0.1,
        num_encoder_blocks=2,
        encoder_final_norm=encoder_final_norm,
        num_classes=10,
        head_type=head_type,
    )


def test_vision_transformer_output_shape_pretrain_head():
    vit = _make_vit(head_type="pretrain")
    x = torch.randn(2, 3, 144, 144)
    out = vit(x)
    assert out.shape == (2, 10)


def test_vision_transformer_output_shape_finetune_head():
    vit = _make_vit(head_type="finetune")
    x = torch.randn(2, 3, 144, 144)
    out = vit(x)
    assert out.shape == (2, 10)


def test_gradient_flow_to_parameters_and_input():
    vit = _make_vit(head_type="pretrain")
    x = torch.randn(2, 3, 144, 144, requires_grad=True)

    out = vit(x).sum()
    out.backward()

    assert x.grad is not None
    # At least one parameter should have gradients
    assert any(p.grad is not None for p in vit.parameters() if p.requires_grad)


def test_invalid_image_size_patch_size_raises():
    with pytest.raises(ValueError):
        _ = VisionTransformer(
            image_size=145,  # not divisible by 12
            in_channels=3,
            patch_size=12,
            embed_dim=128,
            strict=True,
            positional_embedding_dropout_p=0.0,
            num_attention_heads=8,
            mha_attention_dropout_p=0.0,
            mha_proj_dropout_p=0.0,
            mlp_ratio=4.0,
            mlp_dropout_p=0.0,
            num_encoder_blocks=2,
            encoder_final_norm=True,
            num_classes=10,
            head_type="finetune",
        )


def test_invalid_head_type_raises():
    with pytest.raises(ValueError):
        _ = _make_vit(head_type="something_else")


def test_strict_mode_raises_on_wrong_input_resolution():
    vit = _make_vit(strict=True)
    x = torch.randn(2, 3, 288, 288)  # wrong size for strict mode
    with pytest.raises(ValueError):
        _ = vit(x)


def test_non_strict_allows_different_resolution_if_divisible_by_patch_size():
    vit = _make_vit(strict=False)
    x = torch.randn(2, 3, 288, 288)  # divisible by patch_size=12
    out = vit(x)
    assert out.shape == (2, 10)


def test_eval_mode_is_deterministic_train_mode_can_be_stochastic_due_to_dropout():
    vit = _make_vit()
    x = torch.randn(2, 3, 144, 144)

    vit.eval()
    y1 = vit(x)
    y2 = vit(x)
    assert torch.allclose(y1, y2)

    vit.train()
    torch.manual_seed(0)
    y3 = vit(x)
    torch.manual_seed(1)
    y4 = vit(x)
    assert not torch.allclose(y3, y4)


def test_encoder_final_norm_flag_changes_encoder_module_type():
    vit_norm = _make_vit(encoder_final_norm=True)
    vit_no = _make_vit(encoder_final_norm=False)

    assert vit_norm.encoder.final_norm is not None
    assert vit_no.encoder.final_norm is None
