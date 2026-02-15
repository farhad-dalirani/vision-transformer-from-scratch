from __future__ import annotations

from torchvision import transforms

from vision_transformer.config.transform import (
    DEFAULT_MEAN,
    DEFAULT_STD,
    TransformConfig,
)


def _random_resized_crop_ops(image_size: int, resize_min: float, resize_max: float) -> list[transforms.Transform]:
    """Creates a RandomResizedCrop transform for training.

    RandomResizedCrop samples a random region of the input image with a
    randomly chosen area and aspect ratio, then resizes that cropped region
    to ``(image_size, image_size)``.

    The sampled crop area is chosen from the range
    ``[resize_min, resize_max]`` as a fraction of the original image area.
    The aspect ratio is also randomly sampled using torchvision's default
    ratio range.

    Args:
        image_size (int): Target spatial size (height and width) after
            cropping and resizing.
        resize_min (float): Minimum fraction of the original image area to
            sample for the random crop.
        resize_max (float): Maximum fraction of the original image area to
            sample for the random crop.

    Returns:
        list[transforms.Transform]: A list containing a single
        ``transforms.RandomResizedCrop`` transform.
    """
    return   [
        transforms.RandomResizedCrop(
            image_size,
            scale=(resize_min, resize_max),
        )
    ]

def _shorter_side_resize_then_center_crop_ops(image_size: int) -> list[transforms.Transform]:
    """Creates resize + center crop operations for evaluation.

    The image is resized while preserving aspect ratio so that the
    shorter side matches a scaled value (based on 224->256 convention),
    then center-cropped to the target size.

    Args:
        image_size: Target output size.

    Returns:
        A list of torchvision transform ops (resize + center crop).
    """
    resize_size = int(round(image_size  / 224 * 256))
    return [
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
    ]

def _resize_shorter_side(
        image_size: int
    ) -> list[transforms.Transform]:
    """Creates resize-only operations, it does not change input
    image aspect retio.

    Args:
        image_size: Target output size.

    Returns:
        A list of torchvision transform ops (resize only).
    """
    return [transforms.Resize(image_size)]


def build_train_transforms(cfg: TransformConfig) -> transforms.Compose:
    """Builds the training transform pipeline.

    Constructs a torchvision transformation sequence for training images
    based on the provided configuration. The pipeline consists of:

    1. A spatial transform determined by ``cfg.train_resize_policy``:
       - "resize_then_random_crop": Applies RandomResizedCrop to produce
         a square image of size (image_size, image_size).
       - "resize_shorter_side": Resizes the shorter side to ``image_size``
         while preserving aspect ratio (output may not be square).
       - "resize_then_center_crop": Resizes proportionally and then
         center-crops to (image_size, image_size).

    2. Optional data augmentation:
       - Random horizontal flipping (if enabled).
       - RandAugment (if enabled).

    3. Tensor conversion and normalization:
       - Converts PIL image to tensor.
       - Normalizes using either ImageNet statistics or custom
         dataset-specific mean and standard deviation.

    The final output is a square tensor when using
    "resize_then_random_crop" or "resize_then_center_crop". When using
    "resize_shorter_side", the output tensor may retain the original
    aspect ratio.

    Args:
        cfg (TransformConfig): Configuration object specifying spatial
            resizing policy, augmentation parameters, and normalization
            settings.

    Returns:
        transforms.Compose: A composed transform pipeline to be applied
        to training images.

    Raises:
        ValueError: If an unsupported ``train_resize_policy`` is provided.
    """
    mean = DEFAULT_MEAN if cfg.use_default_norm else cfg.mean
    std = DEFAULT_STD if cfg.use_default_norm else cfg.std


    ops = []

    if cfg.train_resize_policy == "resize_then_random_crop":
        spatial_ops = _random_resized_crop_ops(
                image_size=cfg.image_size, 
                resize_min=cfg.train_resize_min, 
                resize_max=cfg.train_resize_max
            )
        ops.extend(spatial_ops)
    elif cfg.train_resize_policy == "resize_shorter_side":
        ops.append(transforms.Resize(cfg.image_size))
    elif cfg.train_resize_policy == "resize_then_center_crop":
        spatial_ops = _shorter_side_resize_then_center_crop_ops(image_size=cfg.image_size)
        ops.extend(spatial_ops)
    else:
        raise ValueError(f"Unknown train_resize_policy: {cfg.train_resize_policy}")

    if cfg.train_horizontal_flip:
        ops.append(transforms.RandomHorizontalFlip(p=0.5))

    if cfg.train_rand_augment:
        ops.append(
            transforms.RandAugment(
                num_ops=cfg.train_rand_augment_num_ops,
                magnitude=cfg.train_rand_augment_magnitude,
            )
        )

    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    return transforms.Compose(ops)




def build_eval_transforms(cfg: TransformConfig) -> transforms.Compose:
    """Builds the evaluation transform pipeline.

    Constructs a deterministic torchvision transformation sequence for
    validation or test images. Unlike the training pipeline, this function
    does not apply stochastic augmentations.

    The spatial transform is determined by ``cfg.eval_resize_policy``:

    - "resize_shorter_side": Resizes the shorter side of the image to
      ``image_size`` while preserving aspect ratio. The resulting image
      may not be square.

    - "resize_then_center_crop": Resizes the image proportionally so that
      the shorter side matches a scaled size (following the 224→256
      convention), then applies a center crop to produce a square image
      of size ``(image_size, image_size)``.

    After spatial resizing, the image is converted to a tensor and
    normalized using either ImageNet statistics or dataset-specific
    mean and standard deviation.

    Args:
        cfg (TransformConfig): Configuration object specifying spatial
            resize policy and normalization parameters.

    Returns:
        transforms.Compose: A composed transform pipeline applied to
        evaluation images.

    Raises:
        ValueError: If an unsupported ``eval_resize_policy`` is provided.
    """
    mean = DEFAULT_MEAN if cfg.use_default_norm else cfg.mean
    std = DEFAULT_STD if cfg.use_default_norm else cfg.std

    if cfg.eval_resize_policy == "resize_shorter_side":
        spatial_ops = _resize_shorter_side(cfg.image_size)
    elif cfg.eval_resize_policy == "resize_then_center_crop":
        spatial_ops = _shorter_side_resize_then_center_crop_ops(image_size=cfg.image_size)
    else:
        raise ValueError(f"Unknown eval_resize_policy: {cfg.eval_resize_policy}")

    return transforms.Compose(
        [
            *spatial_ops,
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
