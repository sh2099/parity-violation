from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


def get_transforms(
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    size: int,
) -> transforms.Compose:
    """Compose the standard transforms."""
    mean_norm = tuple(m / 255.0 for m in mean)
    std_norm = tuple(s / 255.0 for s in std)
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_norm, std=std_norm),
            transforms.Resize(size, antialias=True),
            transforms.CenterCrop(size),
        ]
    )


def load_datasets(
    base_dir: Path,
    test_subdir: str,
    train_subdir: str,
    transforms: transforms.Compose,
) -> Tuple[ImageFolder, ImageFolder]:
    """Load ImageFolder test & train sets."""
    test_path = base_dir / test_subdir
    train_path = base_dir / train_subdir

    test_ds = ImageFolder(root=test_path, transform=transforms)
    train_ds = ImageFolder(root=train_path, transform=transforms)

    return train_ds, test_ds


def flip_dataset(
    dataset: ImageFolder, p: float = 0.5
) -> List[Tuple[torch.Tensor, int]]:
    """Return an in‐memory list of (image, label), randomly flipped."""
    flip = transforms.RandomHorizontalFlip(p=1.0)
    out: List[Tuple[torch.Tensor, int]] = []
    for idx, (img, lbl) in enumerate(dataset):
        if np.random.rand() < p:
            img = flip(img)
        out.append((img, lbl))
    return out


def make_loaders(
    train_ds: ImageFolder,
    test_ds: ImageFolder,
    batch_size: int,
    num_workers: int,
    flip_aug: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Build DataLoader objects, optionally with flipped augmentations."""
    train_data = flip_dataset(train_ds) if flip_aug else train_ds
    test_data = flip_dataset(test_ds) if flip_aug else test_ds

    train_loader = DataLoader(
        train_data, batch_size, shuffle=True, pin_memory=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_data, batch_size, shuffle=False, pin_memory=True, num_workers=num_workers
    )
    return train_loader, test_loader
