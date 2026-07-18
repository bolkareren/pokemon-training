import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHINY_MANIFEST_PATH = PROJECT_ROOT / "shiny_index.json"


class RandomResolutionJitter:
    """Downsample to a random sprite-era resolution, then back up to 224.

    Raw sprites range 56x56 (Gen 1) to 128x128 (Gen 6+) and are all upsampled to
    224x224, so edge thickness and blockiness vary systematically by generation -
    a shortcut the model can key on instead of shape. Randomizing the effective
    source resolution removes it.
    """

    SPRITE_SIZES = (56, 64, 80, 96, 120, 128)

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        if torch.rand(1).item() >= self.p:
            return image
        index = torch.randint(len(self.SPRITE_SIZES), (1,)).item()
        size = self.SPRITE_SIZES[index]
        original = image.size
        return image.resize((size, size), Image.BILINEAR).resize(original, Image.BILINEAR)


class RandomMorphology:
    """Randomly dilate or erode the silhouette by 1-2 pixels.

    Operates on the thresholded binary mask, so it perturbs the contour itself
    rather than greyscale edges. Dilation is a max filter; erosion is the same
    filter on the inverted mask.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, mask):
        if torch.rand(1).item() >= self.p:
            return mask

        kernel = 3 if torch.rand(1).item() < 0.5 else 5
        padding = kernel // 2
        batched = mask.unsqueeze(0)

        if torch.rand(1).item() < 0.5:
            out = torch.nn.functional.max_pool2d(batched, kernel, stride=1, padding=padding)
        else:
            out = -torch.nn.functional.max_pool2d(-batched, kernel, stride=1, padding=padding)

        return out.squeeze(0)


def get_transforms(
    hflip=False,
    morphological=False,
    resolution_jitter=False,
    elastic=False,
):
    """Build (train, eval) transforms. Each augmentation is independently togglable.

    Order matters. Resolution jitter, the affine, elastic and the flip all run on
    the PIL image before tensor conversion; morphology runs after the binary
    threshold, since it is defined on the mask. The threshold sits after the
    geometric augmentations, so any interpolation they introduce is re-binarized
    and the silhouette stays hard-edged.
    """
    geometric = []
    if resolution_jitter:
        geometric.append(RandomResolutionJitter())

    geometric.append(
        transforms.RandomAffine(
            degrees=20,
            translate=(0.2, 0.2),
            scale=(0.85, 1.15),
        )
    )

    if elastic:
        geometric.append(transforms.ElasticTransform(alpha=40.0, sigma=5.0))
    if hflip:
        geometric.append(transforms.RandomHorizontalFlip(p=0.5))

    mask_ops = [RandomMorphology()] if morphological else []

    train_transform = transforms.Compose(
        [
            *geometric,
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > 0.5).float()),
            *mask_ops,
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


def _normal_sprite_indices(dataset):
    """Indices of `dataset` whose image is a normal (non-shiny) sprite.

    Each class's images run as a normal sprite series (one per generation)
    followed by a shiny series repeating those same generations. Shiny sprites
    are recolours, so their silhouettes duplicate the normal ones - keeping both
    puts pixel-identical twins on either side of the train/val split, which made
    ~62% of validation a memorization test. See scripts/duplicate_audit.py.

    The boundary index per class is precomputed into shiny_index.json, since it
    is only derivable from raw sprite dimensions (see
    scripts/generate_shiny_manifest.py) and varies by class: 8 for 52 Pokemon,
    9 for the other 99.
    """
    if not SHINY_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"{SHINY_MANIFEST_PATH} not found - generate it with "
            "`uv run python scripts/generate_shiny_manifest.py`, or set "
            "exclude_shiny=False to train on the duplicated dataset."
        )

    shiny_start = json.loads(SHINY_MANIFEST_PATH.read_text())

    keep = []
    for i, (path, target) in enumerate(dataset.samples):
        # Match the canonical name exactly. data/ can contain strays from earlier
        # processing runs under other conventions (e.g. silhouette-image-1.png,
        # a byte-identical copy of image-1.png); a loose search would read those
        # as a second copy of an index and readmit the duplicate we are removing.
        match = re.fullmatch(r"image-(\d+)\.png", Path(path).name)
        if match is None:
            continue

        class_name = dataset.classes[target]
        if int(match.group(1)) < shiny_start[class_name]:
            keep.append(i)

    return np.array(keep)


def near_duplicate_groups(dataset, indices, iou_threshold=0.97):
    """Group id per index, clustering near-identical silhouettes within a class.

    Excluding shiny sprites removes all exact duplicates but leaves ~4.4%
    near-duplicates: consecutive generations whose sprite art barely changed.
    Grouping them keeps a cluster from straddling a fold boundary, which would
    reintroduce a smaller version of the Phase 11 leak.

    Single-linkage by IoU, computed only within a class - cross-class pairs are
    never grouped, since two different Pokemon sharing a silhouette is a fact
    about the task, not redundancy to be split around.
    """
    targets = np.array(dataset.targets)
    silhouettes = {
        i: np.array(Image.open(dataset.samples[i][0]).convert("L")) > 127 for i in indices
    }

    groups = np.empty(len(indices), dtype=int)
    next_group = 0

    for label in np.unique(targets[indices]):
        members = [(position, i) for position, i in enumerate(indices) if targets[i] == label]

        representatives = []  # (group_id, silhouette) for each cluster so far
        for position, i in members:
            silhouette = silhouettes[i]
            match = None
            for group_id, representative in representatives:
                union = (silhouette | representative).sum()
                if union and (silhouette & representative).sum() / union > iou_threshold:
                    match = group_id
                    break

            if match is None:
                match = next_group
                next_group += 1
                representatives.append((match, silhouette))

            groups[position] = match

    return groups


def fold_indices(dataset, indices, folds, random_state=42, group_aware=True):
    """Yield (train_idx, val_idx) per fold, as indices addressing `dataset`.

    Stratified by class and grouped by near-duplicate cluster. With
    group_aware=False every image is its own group, which reduces this to plain
    stratified K-fold.
    """
    indices = np.asarray(indices)
    targets = np.array(dataset.targets)[indices]
    groups = (
        near_duplicate_groups(dataset, indices) if group_aware else np.arange(len(indices))
    )

    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=random_state)
    for train_positions, val_positions in splitter.split(indices, targets, groups):
        yield indices[train_positions], indices[val_positions]


def split_dataset_indices(dataset, val_size, test_size, random_state=42, indices=None):
    """Stratified train/val/test split, optionally over a subset of `dataset`.

    `indices` restricts the split to a subset (e.g. normal sprites only); the
    returned indices still address the full dataset.
    """
    targets = np.array(dataset.targets)
    indices = np.arange(len(dataset)) if indices is None else np.asarray(indices)
    labels = targets[indices]

    temp_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )

    train_idx, val_idx = train_test_split(
        temp_idx,
        test_size=val_size / (1 - test_size),
        # temp_idx holds dataset-level ids, so index the full target array here,
        # not the subset-aligned `labels`. Identical when indices is None.
        stratify=targets[temp_idx],
        random_state=random_state,
    )

    return train_idx, val_idx, test_idx


def create_datasets(
    data_dir, val_size, test_size, random_state=42, exclude_shiny=True, augmentations=None
):
    train_transform, test_transform = get_transforms(**(augmentations or {}))
    base_dataset = load_dataset(data_dir)
    indices = _normal_sprite_indices(base_dataset) if exclude_shiny else None
    train_idx, val_idx, test_idx = split_dataset_indices(
        base_dataset,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        indices=indices,
    )

    train_dataset = Subset(load_dataset(data_dir, transform=train_transform), train_idx)
    val_dataset = Subset(load_dataset(data_dir, transform=test_transform), val_idx)
    test_dataset = Subset(load_dataset(data_dir, transform=test_transform), test_idx)

    return train_dataset, val_dataset, test_dataset, base_dataset.classes


def create_fold_data_loaders(
    data_dir,
    folds,
    test_size=0.15,
    batch_size=16,
    random_state=42,
    exclude_shiny=True,
    group_aware=True,
    augmentations=None,
):
    """Cross-validation loaders over everything except a held-out test split.

    The test split is carved out *before* cross-validating, so it stays untouched
    across every fold and remains available for a single final evaluation.

    Returns (folds, test_loader, classes) where each fold is
    (train_loader, val_loader, val_idx); val_idx addresses the base dataset so
    out-of-fold predictions can be traced back to individual images.
    """
    train_transform, test_transform = get_transforms(**(augmentations or {}))
    base_dataset = load_dataset(data_dir)
    indices = (
        _normal_sprite_indices(base_dataset)
        if exclude_shiny
        else np.arange(len(base_dataset))
    )
    targets = np.array(base_dataset.targets)

    pool_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=targets[indices],
        random_state=random_state,
    )

    test_loader = DataLoader(
        Subset(load_dataset(data_dir, transform=test_transform), test_idx),
        batch_size=batch_size,
        shuffle=False,
    )

    fold_loaders = []
    for train_idx, val_idx in fold_indices(
        base_dataset, pool_idx, folds, random_state=random_state, group_aware=group_aware
    ):
        shuffle_generator = torch.Generator().manual_seed(random_state)
        train_loader = DataLoader(
            Subset(load_dataset(data_dir, transform=train_transform), train_idx),
            batch_size=batch_size,
            shuffle=True,
            generator=shuffle_generator,
        )
        val_loader = DataLoader(
            Subset(load_dataset(data_dir, transform=test_transform), val_idx),
            batch_size=batch_size,
            shuffle=False,
        )
        fold_loaders.append((train_loader, val_loader, val_idx))

    return fold_loaders, test_loader, base_dataset.classes


def create_data_loaders(
    data_dir,
    val_size=0.1,
    test_size=0.1,
    batch_size=16,
    random_state=42,
    exclude_shiny=True,
    augmentations=None,
):
    train_dataset, val_dataset, test_dataset, classes = create_datasets(
        data_dir=data_dir,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        exclude_shiny=exclude_shiny,
        augmentations=augmentations,
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
