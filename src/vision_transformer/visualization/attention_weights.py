import torch
import torch.nn.functional as F

from vision_transformer.data.utils import sample_batch_from_dataset
from vision_transformer.visualization.utils import (
    heat_to_rgb,
    log_image_grid_to_tensorboard,
    unnormalize_and_upscale,
)


def overlay_heatmap_on_image(
    image: torch.Tensor,
    heat_rgb: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Alpha-blends heatmap on top of image. Both must be (B,3,H,W) in [0,1]."""
    if image.shape != heat_rgb.shape:
        raise ValueError(
            f"image and heatmap must match shape, got {image.shape} vs {heat_rgb.shape}"
        )
    return ((1 - alpha) * image + alpha * heat_rgb).clamp(0.0, 1.0)


def log_model_attention_map(
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
    """Logs CLS-to-patch attention heatmap overlays for a random test batch to
    TensorBoard."""

    xs, _ = sample_batch_from_dataset(test_ds, batch_size=batch_size)

    model.set_mha_blocks_store_attention_flag(True)

    xs = xs.to(device)
    _ = inference_on_one_batch(model=model, batch=xs, device=device)

    xs_vis = unnormalize_and_upscale(
        xs=xs,
        mean=normalization_mean,
        std=normalization_std,
        image_size=image_size,
    )

    # (L) list of (B, H, N, N)
    attn_layers: list[torch.Tensor] = model.get_all_attention_weights()
    if len(attn_layers) == 0:
        raise RuntimeError("No attention weights collected.")

    # Stack layers -> (L, B, H, N, N)
    attn = torch.stack(attn_layers, dim=0)

    # Average over layers and heads -> (B, N, N)
    attn = attn.mean(dim=0).mean(dim=1)

    # CLS -> patches: (B, N-1)
    cls_to_patches = attn[:, 0, 1:]

    # Infer grid size from number of patches
    num_patches = cls_to_patches.shape[1]
    grid_size = model.get_token_grid_size()
    if grid_size * grid_size != num_patches:
        raise ValueError(f"num_patches={num_patches} is not a perfect square.")

    # (B, 1, H, W)
    heat = cls_to_patches.reshape(batch_size, 1, grid_size, grid_size)

    # Normalize per sample to [0, 1]
    heat = heat - heat.amin(dim=(-2, -1), keepdim=True)
    heat = heat / (heat.amax(dim=(-2, -1), keepdim=True) + 1e-8)

    # Upsample to image resolution for nicer viewing
    _, _, H, W = xs_vis.shape
    heat = F.interpolate(heat, size=(H, W), mode="bilinear")
    heat_rgb = heat_to_rgb(heat=heat)

    overlay = overlay_heatmap_on_image(image=xs_vis, heat_rgb=heat_rgb)

    log_image_grid_to_tensorboard(
        writer=writer,
        tag="attention/cls_to_patches",
        images=overlay,
        step=step,
        nrow=8,
    )

    model.set_mha_blocks_store_attention_flag(False)
