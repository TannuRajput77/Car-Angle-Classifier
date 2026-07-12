"""
dataset.py
Loads car images organized in folders by angle class, e.g.:

data/
  train/
    0/
    40/
    90/
    ...
  val/
    0/
    40/
    90/
    ...

Uses torchvision's ImageFolder — simplest way to load classification data
when images are already sorted into class-named subfolders.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random

IMG_SIZE = 128  # reduced from 224 for faster CPU training

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])

    return train_transform, val_transform


def get_dataloaders(data_dir="data", batch_size=32, num_workers=2,
                     max_train_per_class=300, max_val_per_class=75):
    train_transform, val_transform = get_transforms()

    train_dataset = datasets.ImageFolder(f"{data_dir}/train", transform=train_transform)
    val_dataset = datasets.ImageFolder(f"{data_dir}/val", transform=val_transform)

    train_dataset = _subsample_per_class(train_dataset, max_train_per_class)
    val_dataset = _subsample_per_class(val_dataset, max_val_per_class)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    class_names = train_dataset.dataset.classes if isinstance(train_dataset, Subset) else train_dataset.classes
    return train_loader, val_loader, class_names


def _subsample_per_class(dataset, max_per_class):
    """Randomly limit each class to at most max_per_class samples."""
    random.seed(42)
    class_to_indices = {}
    for idx, (_, label) in enumerate(dataset.samples):
        class_to_indices.setdefault(label, []).append(idx)

    selected_indices = []
    for label, indices in class_to_indices.items():
        random.shuffle(indices)
        selected_indices.extend(indices[:max_per_class])

    return Subset(dataset, selected_indices)


if __name__ == "__main__":
    train_loader, val_loader, class_names = get_dataloaders()
    print(f"Classes found: {class_names}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")