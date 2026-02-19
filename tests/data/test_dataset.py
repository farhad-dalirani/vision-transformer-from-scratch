import pytest
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from vision_transformer.config.dataset import DatasetConfig
from vision_transformer.config.transform import TransformConfig
from vision_transformer.data.datasets import build_datasets
from vision_transformer.data.transforms import (
    build_eval_transforms,
    build_train_transforms,
)


@pytest.fixture
def cifar10_datasets(tmp_path):
    cfg_transform = TransformConfig(
        image_size=64, 
        eval_resize_policy="resize_shorter_side"
    )
    train_trans = build_train_transforms(cfg=cfg_transform)
    eval_trans = build_eval_transforms(cfg=cfg_transform)

    ds_conf = DatasetConfig(
        name="cifar10",
        root=tmp_path,
        download=True,
        val_split=0.1,
        split_seed=42,
    )
    return build_datasets(
        cfg=ds_conf, 
        train_transform=train_trans, 
        eval_transform=eval_trans
    )


def test_cifar10_num_samples(cifar10_datasets):
    train_ds, val_ds, test_ds = cifar10_datasets
    for ds in (train_ds, val_ds, test_ds):
        assert isinstance(ds, Dataset)
        assert len(ds) > 0


@pytest.mark.parametrize("which", [0, -1])
def test_cifar10_sample_shape(cifar10_datasets, which):
    train_ds, val_ds, test_ds = cifar10_datasets
    for ds in (train_ds, val_ds, test_ds):
        x, y = ds[which]
        assert x.shape == (3, 64, 64)
        assert y is not None


def test_cifar_10_label_correctness(cifar10_datasets):
    train_ds, val_ds, test_ds = cifar10_datasets
    for ds in (train_ds, val_ds, test_ds):
        for i in [0, 1, 2, -3, -2, -1]:
            _, y = ds[i]
            assert isinstance(y, (int,))
            assert 0 <= int(y) < 10


@pytest.fixture
def cifar100_datasets(tmp_path):
    cfg_transform = TransformConfig(
        image_size=64, 
        eval_resize_policy="resize_shorter_side"
    )
    train_trans = build_train_transforms(cfg=cfg_transform)
    eval_trans = build_eval_transforms(cfg=cfg_transform)

    ds_conf = DatasetConfig(
        name="cifar100",
        root=tmp_path,
        download=True,
        val_split=0.1,
        split_seed=42,
    )
    return build_datasets(
        cfg=ds_conf, 
        train_transform=train_trans, 
        eval_transform=eval_trans
    )

def test_cifar100_num_samples(cifar100_datasets):
    train_ds, val_ds, test_ds = cifar100_datasets
    for ds in (train_ds, val_ds, test_ds):
        assert isinstance(ds, Dataset)
        assert len(ds) > 0

@pytest.mark.parametrize("which", [0, -1])
def test_cifar100_sample_shape(cifar100_datasets, which):
    train_ds, val_ds, test_ds = cifar100_datasets
    for ds in (train_ds, val_ds, test_ds):
        x, y = ds[which]
        assert x.shape == (3, 64, 64)
        assert y is not None


def test_val_split_reasonable(cifar10_datasets):
    train_ds, val_ds, _ = cifar10_datasets
    assert len(val_ds) > 0
    assert len(train_ds) > len(val_ds)


def test_split_deterministic(tmp_path):
    # build twice with same seed/root
    cfg_transform = TransformConfig(
        image_size=64, 
        eval_resize_policy="resize_shorter_side"
    )
    train_trans = build_train_transforms(cfg=cfg_transform)
    eval_trans = build_eval_transforms(cfg=cfg_transform)

    ds_conf = DatasetConfig(
        name="cifar10", 
        root=tmp_path, 
        download=True, 
        val_split=0.1, 
        split_seed=42
    )

    train1, val1, _ = build_datasets(ds_conf, train_trans, eval_trans)
    train2, val2, _ = build_datasets(ds_conf, train_trans, eval_trans)

    assert train1.indices == train2.indices
    assert val1.indices == val2.indices


def test_split_changes_with_seed(tmp_path):
    cfg_transform = TransformConfig(
        image_size=64, 
        eval_resize_policy="resize_shorter_side"
    )
    train_trans = build_train_transforms(cfg=cfg_transform)
    eval_trans = build_eval_transforms(cfg=cfg_transform)

    ds1 = DatasetConfig(
        name="cifar10", 
        root=tmp_path, 
        download=True, 
        val_split=0.1, 
        split_seed=1
    )
    ds2 = DatasetConfig(
        name="cifar10", 
        root=tmp_path, 
        download=True, 
        val_split=0.1, 
        split_seed=2
    )

    train1, val1, _ = build_datasets(ds1, train_trans, eval_trans)
    train2, val2, _ = build_datasets(ds2, train_trans, eval_trans)

    assert train1.indices != train2.indices or val1.indices != val2.indices


def _save_dummy_rgb_image(path, size=(8, 8)):
    img = Image.new("RGB", size=size, color=(123, 222, 64))
    img.save(path)


@pytest.fixture
def imagefolder_root(tmp_path):
    """Creates a tiny ImageFolder dataset with train/val/test splits."""
    root = tmp_path / "dataset"
    classes = ["cat", "dog"]

    for split in ["train", "val", "test"]:
        for cls in classes:
            cls_dir = root / split / cls
            cls_dir.mkdir(parents=True, exist_ok=True)

            # Create 2 images per class per split
            for i in range(2):
                _save_dummy_rgb_image(cls_dir / f"{cls}_{i}.jpg")

    return root


def test_imagefolder_builds_three_splits(imagefolder_root):
    train_tf = transforms.ToTensor()
    eval_tf = transforms.ToTensor()

    cfg = DatasetConfig(name="imagefolder", root=str(imagefolder_root))
    train_ds, val_ds, test_ds = build_datasets(cfg, train_tf, eval_tf)

    # Each split: 2 classes * 2 images = 4
    assert len(train_ds) == 4
    assert len(val_ds) == 4
    assert len(test_ds) == 4

    # Class names should be discovered and consistent across splits
    assert train_ds.classes == ["cat", "dog"]
    assert val_ds.classes == ["cat", "dog"]
    assert test_ds.classes == ["cat", "dog"]


def test_imagefolder_returns_tensor_samples(imagefolder_root):
    train_tf = transforms.ToTensor()
    eval_tf = transforms.ToTensor()

    cfg = DatasetConfig(name="imagefolder", root=str(imagefolder_root))
    train_ds, val_ds, test_ds = build_datasets(cfg, train_tf, eval_tf)

    x, y = train_ds[0]
    assert isinstance(x, torch.Tensor)
    assert x.ndim == 3  # (C, H, W)
    assert x.shape[0] == 3  # RGB
    assert isinstance(y, int)


def test_imagefolder_uses_train_transform_for_train_split(imagefolder_root):
    # Train transform returns zeros tensor; eval returns ones tensor to differentiate
    def train_tf(img):
        return torch.zeros(3, 8, 8)

    def eval_tf(img):
        return torch.ones(3, 8, 8)

    cfg = DatasetConfig(name="imagefolder", root=str(imagefolder_root))
    train_ds, val_ds, test_ds = build_datasets(cfg, train_tf, eval_tf)

    x_train, _ = train_ds[0]
    x_val, _ = val_ds[0]
    x_test, _ = test_ds[0]

    assert torch.all(x_train == 0)
    assert torch.all(x_val == 1)
    assert torch.all(x_test == 1)


def test_imagefolder_custom_split_dir_names(imagefolder_root, tmp_path):
    # Recreate dataset with custom split folder names
    root = tmp_path / "dataset2"
    for split in ["training", "validation", "testing"]:
        for cls in ["a", "b"]:
            d = root / split / cls
            d.mkdir(parents=True, exist_ok=True)
            _save_dummy_rgb_image(d / "x.jpg")

    cfg = DatasetConfig(
        name="imagefolder",
        root=str(root),
        train_dir="training",
        val_dir="validation",
        test_dir="testing",
    )
    train_ds, val_ds, test_ds = build_datasets(
        cfg, transforms.ToTensor(), transforms.ToTensor()
    )

    assert len(train_ds) == 2  # 2 classes * 1 image
    assert len(val_ds) == 2
    assert len(test_ds) == 2


def test_imagefolder_missing_split_raises(tmp_path):
    # Only create train split; val/test missing
    root = tmp_path / "broken"
    (root / "train" / "cat").mkdir(parents=True, exist_ok=True)
    _save_dummy_rgb_image(root / "train" / "cat" / "cat_0.jpg")

    cfg = DatasetConfig(name="imagefolder", root=str(root))

    # ImageFolder will raise if the directory doesn't exist or is empty.
    with pytest.raises(Exception):
        _ = build_datasets(cfg, transforms.ToTensor(), transforms.ToTensor())