from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

from torchvision import transforms

# Defaults commonly used for pretrained RGB vision backbones.
DEFAULT_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
DEFAULT_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

EvalResizePolicy = Literal["resize_then_crop", "resize_only_ratio_preserving"]


@dataclass(frozen=True)
class TransformConfig:
    """Configuration for dataset transforms.

    Attributes:
        image_size: Final output spatial size (H=W=image_size).
        train_resize_min: Minimum area scale for RandomResizedCrop during
            training. Interpreted as a fraction of the original image area.
        train_resize_max: Maximum area scale for RandomResizedCrop during
            training. Interpreted as a fraction of the original image area.
        use_default_norm: If True, uses DEFAULT_MEAN/DEFAULT_STD (imagenet norm)
            for normalization. If False, uses `mean` and `std`.
        mean: Normalization mean (RGB). Only used if `use_default_norm` is False.
        std: Normalization std (RGB). Only used if `use_default_norm` is False.
        eval_resize_policy: Evaluation spatial transform policy.
            - "resize_then_crop": Resize the shorter side to a computed size
              then center crop to `image_size`. Good for natural images where a
              deterministic crop is desired  (e.g. ImageNet).
            - "resize_only_ratio_preserving": Resize directly to `image_size` 
              with no crop. It preserves the aspect ratio. Good for small 
              images or datasets where cropping would remove important 
              content (e.g. CIFAR-10).
    """

    image_size: int = 224
    train_resize_min: float = 0.08
    train_resize_max: float = 1.0
    use_default_norm: bool = True
    mean: Tuple[float, float, float] = DEFAULT_MEAN
    std: Tuple[float, float, float] = DEFAULT_STD
    eval_resize_policy: EvalResizePolicy = "resize_then_crop"


def build_train_transforms(cfg: TransformConfig) -> transforms.Compose:
    """Builds standard training transforms.

    The recipe is:
      RandomResizedCrop -> RandomHorizontalFlip -> ToTensor -> Normalize

    Args:
        cfg: Transform configuration.

    Returns:
        A torchvision.transforms.Compose object for training.
    """
    mean = DEFAULT_MEAN if cfg.use_default_norm else cfg.mean
    std = DEFAULT_STD if cfg.use_default_norm else cfg.std

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                cfg.image_size,
                scale=(cfg.train_resize_min, cfg.train_resize_max),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _eval_resize_then_crop_ops(image_size: int) -> list[transforms.Transform]:
    """Creates resize-then-center-crop operations for evaluation.

    This uses a proportional rule so the crop fraction stays consistent across
    different target resolutions. For image_size=224, it resizes the shorter
    side to 256 and then center crops to 224.

    Args:
        image_size: Target output size.

    Returns:
        A list of torchvision transform ops (resize then crop).
    """
    # Keep the classic 256->224 ratio, scaled to the target image_size.
    resize_size = int(round(image_size / 224 * 256))
    return [
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
    ]


def _eval_resize_only_ratio_preserving_ops(image_size: int) -> list[transforms.Transform]:
    """Creates resize-only operations for evaluation.

    Args:
        image_size: Target output size.

    Returns:
        A list of torchvision transform ops (resize only).
    """
    return [transforms.Resize(image_size)]


def build_eval_transforms(cfg: TransformConfig) -> transforms.Compose:
    """Builds standard evaluation transforms.

    Policies:
      - resize_then_crop: Resize shorter side proportionally, then center crop.
      - resize_only_ratio_preserving: Direct resize to image_size with no crop.

    Args:
        cfg: Transform configuration.

    Returns:
        A torchvision.transforms.Compose object for evaluation.

    Raises:
        ValueError: If cfg.eval_resize_policy is unknown.
    """
    mean = DEFAULT_MEAN if cfg.use_default_norm else cfg.mean
    std = DEFAULT_STD if cfg.use_default_norm else cfg.std

    if cfg.eval_resize_policy == "resize_then_crop":
        spatial_ops = _eval_resize_then_crop_ops(cfg.image_size)
    elif cfg.eval_resize_policy == "resize_only_ratio_preserving":
        spatial_ops = _eval_resize_only_ratio_preserving_ops(cfg.image_size)
    else:
        raise ValueError(f"Unknown eval_resize_policy: {cfg.eval_resize_policy}")

    return transforms.Compose(
        [
            *spatial_ops,
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
