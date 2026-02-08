from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from vision_transformer.config.experiment import ExperimentConfig
from vision_transformer.config.utils import dataclass_from_dict


def save_checkpoint(
    path: str | Path,
    *,
    epoch: int,
    optimizer_step: int,
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    lr_scheduler: Any,
    experiment_config: ExperimentConfig,
    best_val_acc: float | None = None,
) -> None:
    """Save a full training checkpoint to disk.

    This function serializes all state required to fully resume training,
    including model parameters, optimizer state, learning rate scheduler
    state, and the experiment configuration.

    The experiment configuration is stored as a dictionary converted from
    the ``ExperimentConfig`` dataclass and can be reconstructed when loading
    the checkpoint.

    Args:
        path: File path where the checkpoint will be saved.
        epoch: Current training epoch.
        optimizer_step: Number of optimizer update steps completed so far.
        model: Model whose parameters will be saved.
        optim: Optimizer associated with the model.
        lr_scheduler: Learning rate scheduler. May be ``None``.
        experiment_config: Experiment configuration dataclass instance.
        best_val_acc: Best validation accuracy achieved so far, if available.

    Returns:
        None
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "epoch": epoch,
        "optimizer_step": optimizer_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "lr_scheduler_state_dict": (
            lr_scheduler.state_dict() if lr_scheduler is not None else None
        ),
        "experiment_config": asdict(experiment_config),
        "best_val_acc": best_val_acc,
        "torch_version": str(torch.__version__),
    }
    torch.save(ckpt, str(path))


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optim: torch.optim.Optimizer | None = None,
    lr_scheduler: Any | None = None,
    map_location: str | torch.device = "cpu",
):
    """Load a training checkpoint and restore training state.

    This function restores model parameters and, if provided, optimizer and
    learning rate scheduler states from a checkpoint file. It also rebuilds
    the ``ExperimentConfig`` object stored in the checkpoint.

    The caller is responsible for constructing the model, optimizer, and
    scheduler instances before calling this function.

    Args:
        path: Path to the checkpoint file.
        model: Model instance to load parameters into.
        optim: Optimizer instance to restore state into, if resuming training.
        lr_scheduler: Learning rate scheduler to restore state into, if used.
        map_location: Device mapping for loading the checkpoint (e.g. ``"cpu"``
            or ``torch.device("cuda")``).

    Returns:
        dict: A dictionary containing:
            - ``checkpoint``: Raw checkpoint dictionary loaded from disk.
            - ``experiment_config``: Reconstructed ``ExperimentConfig`` object.
            - ``epoch``: Epoch at which the checkpoint was saved.
            - ``optimizer_step``: Optimizer step count at save time.
            - ``best_val_acc``: Best validation accuracy stored in the checkpoint.
    """
    ckpt = torch.load(str(path), map_location=map_location)

    # Restore model
    model.load_state_dict(ckpt["model_state_dict"])

    # Restore optimizer (optional)
    if optim is not None and "optimizer_state_dict" in ckpt:
        optim.load_state_dict(ckpt["optimizer_state_dict"])

    # Restore scheduler (optional)
    if lr_scheduler is not None and ckpt.get("lr_scheduler_state_dict") is not None:
        lr_scheduler.load_state_dict(ckpt["lr_scheduler_state_dict"])

    # Rebuild experiment config object
    experiment_config = dataclass_from_dict(ExperimentConfig, ckpt["experiment_config"])

    return {
        "checkpoint": ckpt,
        "experiment_config": experiment_config,
        "epoch": ckpt.get("epoch", 0),
        "optimizer_step": ckpt.get("optimizer_step", 0),
        "best_val_acc": ckpt.get("best_val_acc", None),
    }
