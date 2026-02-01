from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

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
