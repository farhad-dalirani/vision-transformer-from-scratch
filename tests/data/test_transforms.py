from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from PIL import Image
from torchvision import transforms as tvt

from vision_transformer.data.transforms import (
    DEFAULT_MEAN,
    DEFAULT_STD,
    TransformConfig,
    build_eval_transforms,
    build_train_transforms,
)


def _make_rgb_pil(w: int, h: int) -> Image.Image:
    """Creates a deterministic RGB PIL image.

    Args:
        w: Width in pixels.
        h: Height in pixels.

    Returns:
        A PIL.Image in RGB mode.
    """
    # Deterministic pseudo-image: simple gradient pattern.
    x = torch.linspace(0, 255, steps=w, dtype=torch.uint8).repeat(h, 1)
    y = torch.linspace(0, 255, steps=h, dtype=torch.uint8).unsqueeze(1).repeat(1, w)
    r = x
    g = y
    b = (x // 2 + y // 2).to(torch.uint8)
    arr = torch.stack([r, g, b], dim=-1).numpy()  # (H, W, 3) uint8
    return Image.fromarray(arr, mode="RGB")


def _assert_has_ops(compose: tvt.Compose, expected_types: tuple[type, ...]) -> None:
    """Asserts that a Compose contains ops with the expected types in order.

    Args:
        compose: torchvision.transforms.Compose instance.
        expected_types: Tuple of transform classes expected in order.
    """
    actual = tuple(type(op) for op in compose.transforms)
    assert actual == expected_types


def test_build_train_transforms_has_expected_ops() -> None:
    cfg = TransformConfig(image_size=224)
    t = build_train_transforms(cfg)

    _assert_has_ops(
        t,
        (
            tvt.RandomResizedCrop,
            tvt.RandomHorizontalFlip,
            tvt.ToTensor,
            tvt.Normalize,
        ),
    )

    # Check RRC scale bounds are wired through.
    rrc = t.transforms[0]
    assert rrc.size == (224, 224)
    assert rrc.scale == (cfg.train_resize_min, cfg.train_resize_max)


@pytest.mark.parametrize(
    "policy,expected_first_two",
    [
        (
            "resize_then_crop",
            (tvt.Resize, tvt.CenterCrop),
        ),
        (
            "resize_only_ratio_preserving",
            (tvt.Resize, tvt.ToTensor),  # only one spatial op then ToTensor
        ),
    ],
)
def test_build_eval_transforms_op_sequence(
    policy: str, 
    expected_first_two: tuple[type, type]
) -> None:
    cfg = TransformConfig(image_size=224, eval_resize_policy=policy)
    t = build_eval_transforms(cfg)

    # For resize_then_crop: Resize, CenterCrop, ToTensor, Normalize
    # For resize_only_ratio_preserving: Resize, ToTensor, Normalize
    if policy == "resize_then_crop":
        _assert_has_ops(
            t,
            (tvt.Resize, tvt.CenterCrop, tvt.ToTensor, tvt.Normalize),
        )
    else:
        _assert_has_ops(
            t,
            (tvt.Resize, tvt.ToTensor, tvt.Normalize),
        )


def test_build_eval_transforms_resize_then_crop_computes_expected_resize_size() -> None:
    cfg = TransformConfig(image_size=512, eval_resize_policy="resize_then_crop")
    t = build_eval_transforms(cfg)

    resize = t.transforms[0]
    assert isinstance(resize, tvt.Resize)

    # Expected: round(512/224 * 256) = round(585.142...) = 585
    # torchvision Resize stores size as int when passed an int.
    assert resize.size == 585

    crop = t.transforms[1]
    assert isinstance(crop, tvt.CenterCrop)
    assert crop.size == (512, 512)


def test_build_eval_transforms_resize_only_ratio_preserving_uses_direct_size() -> None:
    cfg = TransformConfig(
        image_size=32, 
        eval_resize_policy="resize_only_ratio_preserving"
    )
    t = build_eval_transforms(cfg)

    resize = t.transforms[0]
    assert isinstance(resize, tvt.Resize)
    assert resize.size == 32


def test_unknown_eval_policy_raises_value_error() -> None:
    cfg = TransformConfig(image_size=224)
    # dataclass is frozen; use dataclasses.replace to modify.
    bad_cfg = replace(cfg, eval_resize_policy="nope")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown eval_resize_policy"):
        build_eval_transforms(bad_cfg)


def test_normalize_parameters_default_norm() -> None:
    cfg = TransformConfig(use_default_norm=True)
    t = build_eval_transforms(cfg)
    norm = t.transforms[-1]
    assert isinstance(norm, tvt.Normalize)
    assert tuple(norm.mean) == DEFAULT_MEAN
    assert tuple(norm.std) == DEFAULT_STD


def test_normalize_parameters_custom_norm() -> None:
    mean = (0.1, 0.2, 0.3)
    std = (0.9, 0.8, 0.7)
    cfg = TransformConfig(use_default_norm=False, mean=mean, std=std)
    t = build_train_transforms(cfg)
    norm = t.transforms[-1]
    assert isinstance(norm, tvt.Normalize)
    assert tuple(norm.mean) == mean
    assert tuple(norm.std) == std


@pytest.mark.parametrize(
    "policy,image_size,input_wh",
    [
        ("resize_then_crop", 224, (400, 400)),
        ("resize_then_crop", 512, (900, 900)),
        ("resize_only_ratio_preserving", 32, (40, 40)),
        ("resize_only_ratio_preserving", 224, (320, 320)),
    ],
)
def test_eval_transform_output_shape(
    policy: str, 
    image_size: int, 
    input_wh: tuple[int, int]
) -> None:
    cfg = TransformConfig(image_size=image_size, eval_resize_policy=policy)
    t = build_eval_transforms(cfg)

    w, h = input_wh
    img = _make_rgb_pil(w, h)
    out = t(img)

    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.float32
    assert out.shape == (3, image_size, image_size)


def test_train_transform_output_shape_is_image_size() -> None:
    # Training includes random crop; output should still match image_size.
    cfg = TransformConfig(image_size=224)
    t = build_train_transforms(cfg)

    img = _make_rgb_pil(640, 480)
    out = t(img)

    assert out.shape == (3, 224, 224)