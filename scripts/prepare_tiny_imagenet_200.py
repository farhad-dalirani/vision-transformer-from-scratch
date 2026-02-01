#!/usr/bin/env python3
"""
Prepare Tiny ImageNet-200 so that train/val/test are all compatible with
torchvision.datasets.ImageFolder.

Given a root like:
~/codes/vision-transformer-from-scratch/datasets/tiny-imagenet-200

This script will:
1) TRAIN: move (or copy) images from train/<wnid>/images/*.JPEG into train/<wnid>/
2) VAL: use val/val_annotations.txt to move (or copy) val/images/*.JPEG into val/<wnid>/
3) TEST: Tiny ImageNet test has no labels. To make it ImageFolder-compatible,
         it puts all test/images/*.JPEG into test/unknown/

By default it MOVES files (fast, no extra disk). Use --copy to keep originals.

Usage:
  python prepare_tiny_imagenet_imagefolder.py \
    --root ~/codes/vision-transformer-from-scratch/datasets/tiny-imagenet-200
"""

import argparse
import os
import shutil
from pathlib import Path


def expand_user(p: str) -> Path:
    return Path(os.path.expanduser(p)).resolve()


def maybe_transfer(src: Path, dst: Path, copy: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return  # already there
    if not src.exists():
        return  # nothing to do
    if copy:
        shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))


def prepare_train(root: Path, copy: bool):
    train_dir = root / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Missing train dir: {train_dir}")

    # Each class: train/<wnid>/images/*.JPEG -> train/<wnid>/*.JPEG
    for wnid_dir in train_dir.iterdir():
        if not wnid_dir.is_dir():
            continue
        images_dir = wnid_dir / "images"
        if not images_dir.exists():
            # maybe already flattened
            continue

        for img in images_dir.glob("*"):
            if img.is_file():
                dst = wnid_dir / img.name
                maybe_transfer(img, dst, copy=copy)

        # remove images_dir if empty
        try:
            if images_dir.exists() and not any(images_dir.iterdir()):
                images_dir.rmdir()
        except OSError:
            pass


def parse_val_annotations(val_annotations: Path):
    """
    Each line:
      <image_name>\t<wnid>\t<x0>\t<y0>\t<x1>\t<y1>
    """
    mapping = {}
    with val_annotations.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            img_name, wnid = parts[0], parts[1]
            mapping[img_name] = wnid
    return mapping


def prepare_val(root: Path, copy: bool):
    val_dir = root / "val"
    if not val_dir.exists():
        raise FileNotFoundError(f"Missing val dir: {val_dir}")

    ann_path = val_dir / "val_annotations.txt"
    if not ann_path.exists():
        # already prepared or nonstandard;
        return

    images_dir = val_dir / "images"
    if not images_dir.exists():
        return

    mapping = parse_val_annotations(ann_path)

    for img_name, wnid in mapping.items():
        src = images_dir / img_name
        dst = val_dir / wnid / img_name
        maybe_transfer(src, dst, copy=copy)

    # remove images_dir if empty
    try:
        if images_dir.exists() and not any(images_dir.iterdir()):
            images_dir.rmdir()
    except OSError:
        pass


def prepare_test(root: Path, copy: bool):
    test_dir = root / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Missing test dir: {test_dir}")

    images_dir = test_dir / "images"
    unknown_dir = test_dir / "unknown"
    unknown_dir.mkdir(parents=True, exist_ok=True)

    if images_dir.exists():
        # Move/copy test/images/*.JPEG -> test/unknown/*.JPEG
        for img in images_dir.glob("*"):
            if img.is_file():
                dst = unknown_dir / img.name
                maybe_transfer(img, dst, copy=copy)

        # remove images_dir if empty
        try:
            if images_dir.exists() and not any(images_dir.iterdir()):
                images_dir.rmdir()
        except OSError:
            pass
    else:
        pass


def verify_imagefolder_compat(split_dir: Path) -> bool:
    """
    ImageFolder expects:
      split_dir/<class_name>/<img files...>
    So: split_dir must contain at least one subdir with at least one file.
    """
    if not split_dir.exists():
        return False
    class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
    if not class_dirs:
        return False
    for d in class_dirs:
        # any file inside (non-recursive) counts
        if any(p.is_file() for p in d.iterdir()):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        required=True,
        help="Path to tiny-imagenet-200 (e.g. ~/codes/.../datasets/tiny-imagenet-200)",
    )
    ap.add_argument(
        "--copy",
        action="store_true",
        help="Copy instead of move (uses more disk, preserves original layout).",
    )
    args = ap.parse_args()

    root = expand_user(args.root)
    print(f"Root: {root}")
    print(f"Mode: {'COPY' if args.copy else 'MOVE'}")

    prepare_train(root, copy=args.copy)
    prepare_val(root, copy=args.copy)
    prepare_test(root, copy=args.copy)

    # Simple checks
    ok_train = verify_imagefolder_compat(root / "train")
    ok_val = verify_imagefolder_compat(root / "val")
    ok_test = verify_imagefolder_compat(root / "test")

    print("\nImageFolder compatibility:")
    print(f"  train: {'OK' if ok_train else 'NOT OK'}  ({root/'train'})")
    print(f"  val:   {'OK' if ok_val else 'NOT OK'}  ({root/'val'})")
    print(f"  test:  {'OK' if ok_test else 'NOT OK'}  ({root/'test'})")

    if not (ok_train and ok_val and ok_test):
        print("\nNote: If val/test were already modified, this script may skip steps.")
        print("If something is NOT OK, inspect the folder structure after running.")


if __name__ == "__main__":
    main()
