import torch


def sample_batch_from_dataset(
    dataset,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly samples a batch of images and labels from a dataset.

    This function selects ``batch_size`` random indices from the given
    PyTorch dataset and returns the corresponding images and labels as
    stacked tensors.

    Args:
        dataset: A PyTorch-compatible dataset object implementing
            ``__len__`` and ``__getitem__``. Each item is expected to
            return a tuple of ``(image, label)``.
        batch_size (int): Number of samples to draw.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - xs: A tensor of shape ``(B, C, H, W)`` containing the
              sampled images.
            - ys: A tensor of shape ``(B,)`` containing the
              corresponding labels.

    Note:
        The sampling is performed without replacement.
    """
    idxs = torch.randperm(len(dataset))[:batch_size]
    xs = torch.stack([dataset[i][0] for i in idxs])  # (B, C, H, W)
    ys = torch.tensor([dataset[i][1] for i in idxs])  # (B,)
    return xs, ys
