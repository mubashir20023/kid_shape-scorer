"""
CutMix Data Augmentation
Reference: Yun et al., "CutMix: Training Strategy that Makes Use of Sample Pairings",
           ICCV 2019.

A rectangular patch is cut from one training image and pasted onto another.
Labels are mixed proportionally to the patch area:
    y_mixed = λ * y_a + (1 - λ) * y_b
where λ = 1 - (patch_area / total_area).
"""

import math
import random
import numpy as np
import torch


def rand_bbox(size: tuple, lam: float):
    """
    Sample a random bounding box for CutMix.

    Parameters
    ----------
    size : (B, C, H, W)
    lam  : mixing ratio sampled from Beta distribution

    Returns
    -------
    (bbx1, bby1, bbx2, bby2) : box corners
    cut_lam                   : true λ after box clipping
    """
    W = size[3]
    H = size[2]

    cut_rat = math.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = random.randint(0, W)
    cy = random.randint(0, H)

    bbx1 = max(0, cx - cut_w // 2)
    bby1 = max(0, cy - cut_h // 2)
    bbx2 = min(W, cx + cut_w // 2)
    bby2 = min(H, cy + cut_h // 2)

    # Recompute λ from actual box area
    cut_lam = 1.0 - (bbx2 - bbx1) * (bby2 - bby1) / (W * H)
    return bbx1, bby1, bbx2, bby2, cut_lam


def cutmix_data(
    images: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 1.0,
) -> tuple:
    """
    Apply CutMix to a batch.

    Parameters
    ----------
    images  : (B, C, H, W) float tensor in [0, 1]
    targets : (B,) long tensor of class indices
    alpha   : Beta distribution parameter

    Returns
    -------
    mixed_images : (B, C, H, W)
    targets_a    : (B,) original labels
    targets_b    : (B,) permuted labels
    lam          : scalar mixing coefficient
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    perm = torch.randperm(batch_size, device=images.device)

    bbx1, bby1, bbx2, bby2, lam = rand_bbox(images.size(), lam)

    mixed = images.clone()
    mixed[:, :, bby1:bby2, bbx1:bbx2] = images[perm, :, bby1:bby2, bbx1:bbx2]

    return mixed, targets, targets[perm], lam


def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    """Compute the CutMix loss."""
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)
