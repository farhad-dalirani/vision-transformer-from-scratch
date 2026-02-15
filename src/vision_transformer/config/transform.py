from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

# Defaults commonly used for pretrained RGB vision backbones.
DEFAULT_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
DEFAULT_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

TransformResizePolicy = Literal[
    "resize_then_random_crop", "resize_shorter_side", "resize_then_center_crop"
]


@dataclass(frozen=True)
class TransformConfig:
    """Configuration for image preprocessing and augmentation.

    Defines spatial resizing strategies, data augmentation options,
    and normalization parameters used for training and evaluation
    pipelines.

    This configuration is consumed by ``build_train_transforms`` and
    ``build_eval_transforms`` to construct deterministic or stochastic
    torchvision transform pipelines.

    Attributes:
        image_size (int):
            Target spatial size. When square-producing policies are used,
            the final output shape will be (image_size, image_size).

        train_resize_policy (TransformResizePolicy):
            Spatial transform policy applied during training. One of:
            - "resize_then_random_crop": Applies RandomResizedCrop,
              producing a square output.
            - "resize_shorter_side": Resizes the shorter side to
              ``image_size`` while preserving aspect ratio.
            - "resize_then_center_crop": Resizes proportionally and
              center-crops to a square output.

        train_resize_min (float):
            Minimum fraction of the original image area sampled when
            using "resize_then_random_crop".

        train_resize_max (float):
            Maximum fraction of the original image area sampled when
            using "resize_then_random_crop".

        train_horizontal_flip (bool):
            If True, applies RandomHorizontalFlip during training.

        train_rand_augment (bool):
            If True, applies RandAugment during training.

        train_rand_augment_num_ops (int):
            Number of augmentation operations sampled per image when
            RandAugment is enabled.

        train_rand_augment_magnitude (int):
            Magnitude parameter controlling the strength of RandAugment
            transformations.

        use_default_norm (bool):
            If True, uses ImageNet normalization statistics
            (DEFAULT_MEAN and DEFAULT_STD). If False, uses ``mean`` and ``std``.

        mean (Tuple[float, float, float]):
            Per-channel normalization mean (RGB), used when
            ``use_default_norm`` is False.

        std (Tuple[float, float, float]):
            Per-channel normalization standard deviation (RGB),
            used when ``use_default_norm`` is False.

        eval_resize_policy (TransformResizePolicy):
            Spatial transform policy applied during evaluation.
            Typically deterministic (e.g., center crop).
            - "resize_then_center_crop": Applies RandomResizedCrop,
              producing a square output.
            - "resize_shorter_side": Resizes the image proportionally
              so that the shorter side matches a scaled size, then applies
              a center crop to produce a square image
    """

    image_size: int = 224

    train_resize_policy: TransformResizePolicy = "resize_then_random_crop"

    # Use if train resize policy is "resize_then_random_crop"
    train_resize_min: float = 0.3
    train_resize_max: float = 1.0

    train_horizontal_flip: bool = True

    train_rand_augment: bool = True
    train_rand_augment_num_ops: int = 2
    train_rand_augment_magnitude: int = 9

    use_default_norm: bool = True
    mean: Tuple[float, float, float] = DEFAULT_MEAN
    std: Tuple[float, float, float] = DEFAULT_STD

    eval_resize_policy: TransformResizePolicy = "resize_then_center_crop"
