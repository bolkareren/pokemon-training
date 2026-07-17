import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_transforms():
    train_transform = transforms.Compose(
        [
            transforms.RandomAffine(
                degrees=20,
                translate=(0.2, 0.2),
                scale=(0.85, 1.15),
            ),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > 0.5).float()),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > 0.5).float()),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return train_transform, test_transform


def load_dataset(data_dir="data", transform=None):
    return datasets.ImageFolder(data_dir, transform=transform)


def split_dataset_indices(dataset, val_size, test_size, random_state=42):
    labels = np.array(dataset.targets)
    indices = np.arange(len(dataset))

    temp_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )

    train_idx, val_idx = train_test_split(
        temp_idx,
        test_size=val_size / (1 - test_size),
        stratify=labels[temp_idx],
        random_state=random_state,
    )

    return train_idx, val_idx, test_idx


def create_datasets(data_dir, val_size, test_size, random_state=42):
    train_transform, test_transform = get_transforms()
    base_dataset = load_dataset(data_dir)
    train_idx, val_idx, test_idx = split_dataset_indices(
        base_dataset,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
    )

    train_dataset = Subset(load_dataset(data_dir, transform=train_transform), train_idx)
    val_dataset = Subset(load_dataset(data_dir, transform=test_transform), val_idx)
    test_dataset = Subset(load_dataset(data_dir, transform=test_transform), test_idx)

    return train_dataset, val_dataset, test_dataset, base_dataset.classes


def create_data_loaders(data_dir, val_size=0.1, test_size=0.1, batch_size=16, random_state=42):
    train_dataset, val_dataset, test_dataset, classes = create_datasets(
        data_dir=data_dir,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
    )

    # Pin the shuffle to its own generator seeded with random_state so epoch
    # ordering is reproducible and independent of global-RNG consumption order.
    shuffle_generator = torch.Generator().manual_seed(random_state)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=shuffle_generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    return train_loader, val_loader, test_loader, classes
