from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import Subset, random_split
from torchvision import datasets
from torchvision.datasets import ImageFolder


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for building train, validation, and test datasets.

    Supports:
      - ImageFolder-style datasets with optional validation directory.
      - Torchvision datasets (e.g., CIFAR-10/100) where validation is created
        by splitting the training set.

    Notes:
        For torchvision datasets like CIFAR-10/100, the directory fields
        (train_dir, val_dir, test_dir) are ignored because torchvision defines
        the splits internally.

    Attributes:
        name: Dataset identifier (e.g., "cifar10", "cifar100", "imagefolder",
            "tinyimagenet-200").
        root: Root directory containing the dataset or download target.
        train_dir: Training split subdirectory name (ImageFolder-style only).
        val_dir: Validation split subdirectory name, or None if no dedicated
            validation directory exists (ImageFolder-style only).
        test_dir: Test split subdirectory name (ImageFolder-style only).
        download: Whether to download the dataset if supported (torchvision
            datasets only, ignored for ImageFolder).
        val_split: Fraction of the training set reserved for validation when
            no official validation split exists (e.g., CIFAR).
        split_seed: Random seed for deterministic train/validation splitting.
    """
    name: str
    root: str

    # ImageFolder-only arguments
    train_dir: str | None = "train"
    val_dir: str | None = "val"
    test_dir: str | None = "test"

    # Torchvision datasets (e.g., CIFAR)
    download: bool = True

    # Train/val split for datasets without official validation
    val_split: float = 0.1
    split_seed: int = 42

def get_num_classes(dataset_name: str) -> int:
    """Returns the number of classes for a supported dataset.

    Args:
        dataset_name: Name of the dataset (e.g., "cifar10", "cifar100",
            "tinyimagenet-200", or "imagefolder").

    Returns:
        Number of classes in the dataset. Returns -1 for datasets where the
        number of classes is determined dynamically from the directory
        structure (e.g., ImageFolder).

    Raises:
        ValueError: If the dataset name is not recognized.
    """
    name = dataset_name.lower()

    if name in {"imagefolder"}:
        return -1
    if name in {"tinyimagenet-200"}:
        return 200
    if name == "cifar10":
        return 10
    if name == "cifar100":
        return 100

    raise ValueError(f"Unknown dataset name: {dataset_name}")

def _validate_val_split(val_split: float) -> None:
    """Validates that the validation split fraction is in the range (0, 1).

    Args:
        val_split: Fraction of the training set reserved for validation.

    Raises:
        ValueError: If the validation split is not strictly between 0 and 1.
    """
    if not (0.0 < val_split < 1.0):
        raise ValueError(f"val_split must be in (0, 1), got {val_split}")


def _split_train_val_indices(
    num_samples: int, val_split: float, seed: int
) -> Tuple[list[int], list[int]]:
    """Deterministically splits dataset indices into train and validation sets.

    Args:
        num_samples: Total number of samples in the dataset.
        val_split: Fraction of samples to assign to the validation set.
        seed: Random seed used to ensure deterministic splitting.

    Returns:
        A tuple containing:
            - train_indices: Indices assigned to the training set.
            - val_indices: Indices assigned to the validation set.
    """
    _validate_val_split(val_split)
    val_size = int(num_samples * val_split)
    train_size = num_samples - val_size

    generator = torch.Generator().manual_seed(seed)
    train_idx, val_idx = random_split(
        range(num_samples), [train_size, val_size], generator=generator
    )
    return list(train_idx.indices), list(val_idx.indices)


def build_datasets(
    cfg: DatasetConfig,
    train_transform,
    eval_transform,
) -> Tuple[object, object, object]:
    """Builds training, validation, and test datasets.

    This function supports both ImageFolder-style datasets and torchvision
    datasets such as CIFAR. If no official validation directory is available
    for ImageFolder-style datasets, the training set is split into training
    and validation subsets.

    Args:
        cfg: Dataset configuration specifying dataset type and paths.
        train_transform: Transform applied to training samples.
        eval_transform: Transform applied to validation and test samples.

    Returns:
        A tuple containing:
            - train_ds: Training dataset.
            - val_ds: Validation dataset.
            - test_ds: Test dataset.

    Raises:
        ValueError: If the dataset name specified in the configuration
            is not supported.
    """
    name = cfg.name.lower()

    # --- ImageFolder-style (ImageNet-like layouts)
    if name in {"tinyimagenet-200", "imagefolder"}:
        train_path = os.path.join(cfg.root, cfg.train_dir)
        test_path = os.path.join(cfg.root, cfg.test_dir)

        # val is optional
        val_path = None
        if cfg.val_dir:  # not None/empty
            candidate = os.path.join(cfg.root, cfg.val_dir)
            if os.path.isdir(candidate):
                val_path = candidate

        # Always load test normally
        test_ds = ImageFolder(root=test_path, transform=eval_transform)

        if val_path is not None:
            # official val folder exists
            train_ds = ImageFolder(root=train_path, transform=train_transform)
            val_ds = ImageFolder(root=val_path, transform=eval_transform)
            return train_ds, val_ds, test_ds

        # No val folder -> split train into train/val (different transforms)
        full_train_for_train = ImageFolder(root=train_path, transform=train_transform)
        full_train_for_val = ImageFolder(root=train_path, transform=eval_transform)

        train_indices, val_indices = _split_train_val_indices(
            num_samples=len(full_train_for_train),
            val_split=cfg.val_split,
            seed=cfg.split_seed,
        )
        train_ds = Subset(full_train_for_train, train_indices)
        val_ds = Subset(full_train_for_val, val_indices)
        return train_ds, val_ds, test_ds

    # --- CIFAR-10
    if name == "cifar10":
        full_train_for_train = datasets.CIFAR10(
            root=cfg.root, train=True, transform=train_transform, download=cfg.download
        )
        full_train_for_val = datasets.CIFAR10(
            root=cfg.root, train=True, transform=eval_transform, download=cfg.download
        )
        test_ds = datasets.CIFAR10(
            root=cfg.root, train=False, transform=eval_transform, download=cfg.download
        )

        train_indices, val_indices = _split_train_val_indices(
            num_samples=len(full_train_for_train),
            val_split=cfg.val_split,
            seed=cfg.split_seed,
        )
        train_ds = Subset(full_train_for_train, train_indices)
        val_ds = Subset(full_train_for_val, val_indices)
        return train_ds, val_ds, test_ds

    # --- CIFAR-100
    if name == "cifar100":
        full_train_for_train = datasets.CIFAR100(
            root=cfg.root, train=True, transform=train_transform, download=cfg.download
        )
        full_train_for_val = datasets.CIFAR100(
            root=cfg.root, train=True, transform=eval_transform, download=cfg.download
        )
        test_ds = datasets.CIFAR100(
            root=cfg.root, train=False, transform=eval_transform, download=cfg.download
        )

        train_indices, val_indices = _split_train_val_indices(
            num_samples=len(full_train_for_train),
            val_split=cfg.val_split,
            seed=cfg.split_seed,
        )
        train_ds = Subset(full_train_for_train, train_indices)
        val_ds = Subset(full_train_for_val, val_indices)
        return train_ds, val_ds, test_ds

    raise ValueError(f"Unknown dataset '{cfg.name}'.")
