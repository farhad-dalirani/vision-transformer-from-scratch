import torch
from PIL import ImageDraw
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from vision_transformer.data.utils import sample_batch_from_dataset
from vision_transformer.visualization.utils import (
    log_image_grid_to_tensorboard,
    unnormalize_and_upscale,
)


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
    log_image_grid_to_tensorboard(
        writer=writer,
        tag="predictions/samples",
        images=annotated,
        step=step,
        nrow=8,
    )
