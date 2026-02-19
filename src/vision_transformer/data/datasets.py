from __future__ import annotations

import os
from typing import Tuple

import torch
from torch.utils.data import Subset, random_split
from torchvision import datasets
from torchvision.datasets import ImageFolder

from vision_transformer.config.dataset import DatasetConfig


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
    if name == "imagefolder":
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

    # --- ImageNet-1K
    if name == "imagenet-1k":
        full_train_for_train = datasets.ImageNet(
            root=cfg.root, train=True, transform=train_transform, download=cfg.download
        )
        full_train_for_val = datasets.ImageNet(
            root=cfg.root, train=True, transform=eval_transform, download=cfg.download
        )
        test_ds = datasets.ImageNet(
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

    # --- German Traffic Sign Recognition Benchmark (GTSRB)
    if name == "gtsrb":
        full_train_for_train = datasets.GTSRB(
            root=cfg.root, split="train", transform=train_transform, download=cfg.download
        )
        full_train_for_val = datasets.GTSRB(
            root=cfg.root, split="train", transform=eval_transform, download=cfg.download
        )
        test_ds = datasets.GTSRB(
            root=cfg.root, split="test", transform=eval_transform, download=cfg.download
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
