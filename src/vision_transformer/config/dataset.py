from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

datasetName = Literal["imagefolder", "imagenet-1k", "cifar10", "cifar100", "gtsrb"]

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
        name: Dataset identifier (e.g., "cifar10", "cifar100", "imagenet-1k", 
            "gtsrb", "imagefolder").
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
    name: datasetName = "cifar10"
    root: str = "./datasets"

    # ImageFolder-only arguments
    train_dir: str | None = "train"
    val_dir: str | None = "val"
    test_dir: str | None = "test"

    # Torchvision datasets (e.g., CIFAR)
    download: bool = True

    # Train/val split for datasets without official validation
    val_split: float = 0.1
    split_seed: int = 42