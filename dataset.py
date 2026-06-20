"""
dataset.py — Dataset loading for the BOT-2 hand-drawn shapes scoring task.
"""

import os
import re
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from config import (
    DATASET_PATH, IMG_SIZE, MIN_CLASS_COUNT, SHAPES, SKIP_FILES,
    VAL_SPLIT, TEST_SPLIT,
)


def get_train_transforms(img_size: int = IMG_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1),
                                scale=(0.85, 1.15)),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])


def get_val_transforms(img_size: int = IMG_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def _get_corner_crop(name: str, size: int):
    if name == "center":
        return transforms.CenterCrop(size)
    elif name == "tl":
        return transforms.Lambda(lambda img: img.crop((0, 0, size, size)))
    elif name == "tr":
        return transforms.Lambda(
            lambda img: img.crop((img.width - size, 0, img.width, size)))
    elif name == "bl":
        return transforms.Lambda(
            lambda img: img.crop((0, img.height - size, size, img.height)))
    elif name == "br":
        return transforms.Lambda(
            lambda img: img.crop(
                (img.width - size, img.height - size, img.width, img.height)))


def get_tta_transforms(img_size: int = IMG_SIZE) -> list:
    tf_list = []
    for hflip in [False, True]:
        for crop_name in ["center", "tl", "tr", "bl", "br"]:
            steps = [transforms.Resize((img_size + 32, img_size + 32))]
            if hflip:
                steps.append(transforms.RandomHorizontalFlip(p=1.0))
            steps.append(_get_corner_crop(crop_name, img_size))
            steps += [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]
            tf_list.append(transforms.Compose(steps))
    return tf_list


class ShapeDataset(Dataset):
    def __init__(self, samples, transform=None,
                 diffusemix=None, dm_prob=0.4, tta=False):
        self.samples    = samples
        self.transform  = transform
        self.diffusemix = diffusemix
        self.dm_prob    = dm_prob
        self.tta        = tta

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")

        if self.diffusemix is not None and torch.rand(1).item() < self.dm_prob:
            image = self.diffusemix(image)

        if self.tta and isinstance(self.transform, list):
            views = torch.stack([tf(image) for tf in self.transform])
            return views, label

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def merge_rare_classes(labels: list, min_count: int = MIN_CLASS_COUNT) -> tuple:
    counts  = Counter(labels)
    all_cls = sorted(counts.keys())
    remap   = {c: c for c in all_cls}

    changed = True
    while changed:
        changed = False
        cur_labels = [remap[l] for l in labels]
        counts     = Counter(cur_labels)
        for c in sorted(set(remap.values())):
            if counts.get(c, 0) < min_count:
                candidates = [
                    x for x in sorted(set(remap.values()))
                    if x != c and counts.get(x, 0) >= min_count
                ]
                if not candidates:
                    continue
                target = min(candidates, key=lambda x: abs(x - c))
                remap  = {k: (target if v == c else v) for k, v in remap.items()}
                warnings.warn(
                    f"Class {c} has only {counts.get(c, 0)} sample(s) — "
                    f"merging into class {target}.", UserWarning
                )
                changed = True

    new_labels = [remap[l] for l in labels]
    return new_labels, remap


def _reindex(labels: list) -> tuple:
    uniq    = sorted(set(labels))
    reindex = {old: new for new, old in enumerate(uniq)}
    return [reindex[l] for l in labels], reindex


def get_class_weights(labels: list, num_classes: int) -> torch.Tensor:
    counts  = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts  = np.where(counts == 0, 1, counts)
    weights = counts.sum() / (num_classes * counts)
    weights = np.clip(weights, 0.1, 10.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


_FILENAME_RE = re.compile(
    r"img\d+-(.+)-(\d+)(?:\([^)]*\))?\.png",
    re.IGNORECASE,
)


def load_shape_samples(shape_name: str) -> list:
    info   = SHAPES[shape_name]
    folder = os.path.join(DATASET_PATH, shape_name)
    if not os.path.isdir(folder):
        for alt in (shape_name.replace(" ", "_"),
                    shape_name.replace("_", " ")):
            cand = os.path.join(DATASET_PATH, alt)
            if os.path.isdir(cand):
                folder = cand
                break

    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"Shape folder not found: {folder}\n"
            f"Check DATASET_PATH in config.py and that the folder name matches "
            f"exactly (including spaces and capitalisation)."
        )

    samples = []
    skipped_parse = 0
    skipped_score = 0

    for fname in os.listdir(folder):
        if fname in SKIP_FILES:
            continue
        m = _FILENAME_RE.match(fname)
        if m is None:
            skipped_parse += 1
            continue
        score = int(m.group(2))
        if score > info["max_score"]:
            skipped_score += 1
            continue
        samples.append((os.path.join(folder, fname), score))

    if skipped_parse > 0:
        warnings.warn(
            f"{shape_name}: {skipped_parse} file(s) did not match the expected "
            f"naming pattern and were skipped.", UserWarning
        )
    if skipped_score > 0:
        warnings.warn(
            f"{shape_name}: {skipped_score} file(s) had scores above "
            f"max_score={info['max_score']} and were skipped.", UserWarning
        )

    return samples


def safe_split(paths, labels, test_size, seed):
    from collections import Counter as _Counter
    counts = _Counter(labels)

    singleton_idx = [i for i, l in enumerate(labels) if counts[l] == 1]
    normal_idx    = [i for i, l in enumerate(labels) if counts[l] > 1]

    if singleton_idx:
        warnings.warn(
            f"{len(singleton_idx)} singleton class(es) found. "
            "Keeping their sample(s) in the training split.", UserWarning
        )

    split_paths  = [paths[i]  for i in normal_idx]
    split_labels = [labels[i] for i in normal_idx]

    singleton_paths  = [paths[i]  for i in singleton_idx]
    singleton_labels = [labels[i] for i in singleton_idx]

    if len(split_paths) == 0:
        return split_paths + singleton_paths, [], split_labels + singleton_labels, []

    normal_counts = _Counter(split_labels)
    can_stratify  = all(v >= 2 for v in normal_counts.values())

    tr_p, te_p, tr_l, te_l = train_test_split(
        split_paths, split_labels,
        test_size=test_size,
        stratify=split_labels if can_stratify else None,
        random_state=seed,
    )

    tr_p = list(tr_p) + singleton_paths
    tr_l = list(tr_l) + singleton_labels

    return tr_p, te_p, tr_l, te_l


def build_dataloaders(
    shape_name: str,
    batch_size: int = 32,
    val_split: float = VAL_SPLIT,
    test_split: float = TEST_SPLIT,
    num_workers: int = 0,
    diffusemix=None,
    seed: int = 42,
    use_tta: bool = False,
) -> tuple:
    """
    Build train / val / test DataLoaders for a single shape.

    Returns
    -------
    train_loader, val_loader, test_loader, num_classes, class_weights, label_remap
    """
    samples = load_shape_samples(shape_name)
    paths   = [s[0] for s in samples]
    labels  = [s[1] for s in samples]

    print(f"  {shape_name}: {len(samples)} images, "
          f"score distribution: {dict(sorted(Counter(labels).items()))}")

    labels, merge_remap = merge_rare_classes(labels, min_count=MIN_CLASS_COUNT)
    labels, reindex_map = _reindex(labels)

    label_remap = {
        orig: reindex_map[merge_remap[orig]]
        for orig in merge_remap
    }

    num_classes   = len(set(labels))
    class_weights = get_class_weights(labels, num_classes)

    train_paths, test_paths, train_labels, test_labels = safe_split(
        paths, labels, test_split, seed
    )
    train_paths, val_paths, train_labels, val_labels = safe_split(
        train_paths, train_labels,
        val_split / (1.0 - test_split), seed
    )

    train_samples = list(zip(train_paths, train_labels))
    val_samples   = list(zip(val_paths,   val_labels))
    test_samples  = list(zip(test_paths,  test_labels))

    train_tf = get_train_transforms()
    val_tf   = get_val_transforms()
    tta_tf   = get_tta_transforms() if use_tta else None

    sample_weights = [class_weights[l].item() for l in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_ds = ShapeDataset(train_samples, train_tf, diffusemix)
    val_ds   = ShapeDataset(val_samples,   val_tf)
    test_ds  = ShapeDataset(
        test_samples,
        tta_tf if use_tta else val_tf,
        tta=use_tta,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=False,
    )
    val_loader   = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )
    test_loader  = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )

    return (train_loader, val_loader, test_loader,
            num_classes, class_weights, label_remap)