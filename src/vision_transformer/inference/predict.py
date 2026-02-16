from typing import Any, Dict

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
from tqdm.auto import tqdm


def evaluate_classifier(
    model: Module,
    dataloader: DataLoader,
    device: str,
    num_classes: int,
    average: str = "macro",
    compute_confusion_matrix: bool = True,
) -> Dict[str, Any]:
    model.eval()

    metrics = MetricCollection(
        {
            "f1": MulticlassF1Score(num_classes=num_classes, average=average),
            "accuracy": MulticlassAccuracy(num_classes=num_classes, average="micro"),
            "precision": MulticlassPrecision(num_classes=num_classes, average=average),
            "recall": MulticlassRecall(num_classes=num_classes, average=average),
            **(
                {"confusion_matrix": MulticlassConfusionMatrix(num_classes=num_classes)}
                if compute_confusion_matrix
                else {}
            ),
        }
    ).to(device)

    metrics.reset()

    with torch.inference_mode():
        for X, y in tqdm(dataloader, desc="Evaluation", leave=True):
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(X)
            preds = torch.argmax(logits, dim=1)

            metrics.update(preds, y)

    out = metrics.compute()

    # Move to CPU only once at the end (still syncs, but far less overhead overall)
    result: Dict[str, Any] = {
        "f1": out["f1"].detach().cpu().item(),
        "accuracy": out["accuracy"].detach().cpu().item(),
        "precision": out["precision"].detach().cpu().item(),
        "recall": out["recall"].detach().cpu().item(),
    }
    if compute_confusion_matrix:
        result["confusion_matrix"] = out["confusion_matrix"].detach().cpu()

    return result


def run_inference(model, dataloader: DataLoader, device: str) -> torch.Tensor:
    """Run inference on a dataset without labels.

    Performs a forward pass of the model over the given dataloader and returns
    the concatenated logits for all samples. No metrics are computed.

    This function is intended for unlabeled datasets or pure inference use cases.

    Args:
        model: A PyTorch model returning logits.
        dataloader: DataLoader yielding batches of inputs.
        device: Device on which inference is performed (e.g. "cpu", "cuda").

    Returns:
        torch.Tensor: Concatenated model outputs (logits) of shape
            (num_samples, num_classes).
    """
    model.eval()
    outputs = []

    with torch.inference_mode():
        for X in tqdm(dataloader, desc="Inference"):
            X = X.to(device)
            logits = model(X)
            outputs.append(logits.cpu())

    return torch.cat(outputs)


def inference_on_one_batch(
    model: torch.nn.Module, batch: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """
    Run inference on a single batch and return predicted class indices.

    Args:
        model (torch.nn.Module): Trained classification model.
        batch (torch.Tensor): Input batch of shape (B, ...).
        device (torch.device): Device to run inference on.

    Returns:
        torch.Tensor: Predicted class indices of shape (B,)
                      returned on CPU.
    """
    model.eval()
    batch = batch.to(device)

    with torch.inference_mode():
        logits = model(batch)
        classes = torch.argmax(logits, dim=1)

    return classes.cpu()
