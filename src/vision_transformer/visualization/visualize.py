import torch
import torch.nn.functional as F
from PIL import ImageDraw
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

from vision_transformer.data.utils import sample_batch_from_dataset


def unnormalize_and_upscale(
    xs: torch.Tensor,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    image_size: int,
    min_vis_size: int = 128,
    upscale_factor: int = 4,
) -> torch.Tensor:
    """Reverses normalization and optionally upscales images for visualization.

    Works for arbitrary channel counts (e.g., 1, 3, 4).
    """

    if xs.ndim != 4:
        raise ValueError(f"Expected input shape (B, C, H, W), got {xs.shape}")

    B, C, H, W = xs.shape

    if len(mean) != C or len(std) != C:
        raise ValueError(
            f"Mean/std length must match number of channels ({C}), "
            f"got mean={len(mean)}, std={len(std)}"
        )

    mean_t = torch.tensor(mean, dtype=xs.dtype, device=xs.device).view(1, C, 1, 1)
    std_t = torch.tensor(std, dtype=xs.dtype, device=xs.device).view(1, C, 1, 1)

    xs_vis = (xs * std_t + mean_t).clamp(0.0, 1.0)

    if image_size < min_vis_size:
        xs_vis = F.interpolate(xs_vis, scale_factor=upscale_factor, mode="nearest")

    return xs_vis


def annotate_images_with_labels(
    xs_vis: torch.Tensor,
    ys: torch.Tensor,
    pred_ys: torch.Tensor,
    text_xy: tuple[int, int] = (2, 2),
) -> torch.Tensor:
    """Overlays GT/pred text on each image and returns a BCHW tensor batch."""
    to_tensor = transforms.ToTensor()
    annotated = []

    for i in range(xs_vis.shape[0]):
        img = to_pil_image(xs_vis[i].cpu())  # PIL image
        draw = ImageDraw.Draw(img)

        gt = int(ys[i])
        pr = int(pred_ys[i])
        draw.text(text_xy, f"GT: {gt} | Pred: {pr}", fill=(255, 255, 255))

        annotated.append(to_tensor(img))

    return torch.stack(annotated)  # (B, C, H, W)


def log_prediction_grid_to_tensorboard(
    writer,
    tag: str,
    images: torch.Tensor,
    step: int,
    nrow: int = 8,
) -> None:
    """Logs a BCHW image batch as a grid to TensorBoard."""
    grid = make_grid(images, nrow=min(nrow, images.shape[0]))
    writer.add_image(tag, grid, global_step=step)


def log_model_predictions(
    *,
    writer,
    model,
    test_ds,
    device: str,
    step: int,
    batch_size: int,
    image_size: int,
    normalization_mean: tuple[float, float, float],
    normalization_std: tuple[float, float, float],
    inference_on_one_batch,
) -> None:
    """Logs a labeled prediction grid for one random test batch."""

    xs, ys = sample_batch_from_dataset(test_ds, batch_size=batch_size)

    # Inference
    pred_ys = inference_on_one_batch(model=model, batch=xs, device=device)  # (B,)

    xs_vis = unnormalize_and_upscale(
        xs=xs,
        mean=normalization_mean,
        std=normalization_std,
        image_size=image_size,
    )

    annotated = annotate_images_with_labels(xs_vis=xs_vis, ys=ys, pred_ys=pred_ys)
    log_prediction_grid_to_tensorboard(
        writer=writer,
        tag="predictions/samples",
        images=annotated,
        step=step,
        nrow=8,
    )
