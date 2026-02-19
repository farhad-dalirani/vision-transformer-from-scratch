import torch
import torch.nn.functional as F
from torchvision.utils import make_grid


def log_image_grid_to_tensorboard(
    writer,
    tag: str,
    images: torch.Tensor,
    step: int,
    nrow: int = 8,
) -> None:
    """Logs a BCHW image batch as a grid to TensorBoard."""
    grid = make_grid(images, nrow=min(nrow, images.shape[0]))
    writer.add_image(tag, grid, global_step=step)


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


def heat_to_rgb(heat: torch.Tensor) -> torch.Tensor:
    """Converts (B,1,H,W) heat in [0,1] to RGB (B,3,H,W) (red channel)."""
    if heat.ndim != 4 or heat.shape[1] != 1:
        raise ValueError(f"heat must be (B,1,H,W), got {tuple(heat.shape)}")
    return torch.cat([heat, torch.zeros_like(heat), torch.zeros_like(heat)], dim=1)
