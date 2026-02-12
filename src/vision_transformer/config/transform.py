from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

# Defaults commonly used for pretrained RGB vision backbones.
DEFAULT_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
DEFAULT_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

TransformResizePolicy = Literal["resize_then_crop", "resize_only_ratio_preserving"]


@dataclass(frozen=True)
class TransformConfig:
    """Configuration for dataset transforms.

    This dataclass defines all parameters controlling spatial transforms,
    augmentation strategies, and normalization applied to images during
    training and evaluation.

    Attributes:
        image_size (int): Final output spatial size (H = W = image_size).

        train_resize_policy (TransformResizePolicy): Spatial transform
            policy used during training. One of:
            - "resize_then_crop": Applies RandomResizedCrop.
            - "resize_only_ratio_preserving": Applies Resize(image_size).

        train_resize_min (float): Minimum scale factor for
            RandomResizedCrop when using "resize_then_crop".
            Interpreted as a fraction of the original image area.

        train_resize_max (float): Maximum scale factor for
            RandomResizedCrop when using "resize_then_crop".

        train_rand_augment (bool): If True, applies RandAugment during
            training.

        train_rand_augment_num_ops (int): Number of augmentation
            operations applied by RandAugment.

        train_rand_augment_magnitude (int): Magnitude parameter for
            RandAugment transformations.

        use_default_norm (bool): If True, uses ImageNet normalization
            constants (DEFAULT_MEAN and DEFAULT_STD).

        mean (Tuple[float, float, float]): Per-channel normalization
            mean used when `use_default_norm` is False.

        std (Tuple[float, float, float]): Per-channel normalization
            standard deviation used when `use_default_norm` is False.

        eval_resize_policy (TransformResizePolicy): Spatial transform
            policy used during evaluation.
    """

    image_size: int = 224

    train_resize_policy: TransformResizePolicy = "resize_then_crop"

    # Use if resize policy is "resize_then_crop"
    train_resize_min: float = 0.3
    train_resize_max: float = 1.0

    train_rand_augment: bool = True
    train_rand_augment_num_ops: int = 2
    train_rand_augment_magnitude: int = 9

    use_default_norm: bool = True
    mean: Tuple[float, float, float] = DEFAULT_MEAN
    std: Tuple[float, float, float] = DEFAULT_STD

    eval_resize_policy: TransformResizePolicy = "resize_then_crop"
