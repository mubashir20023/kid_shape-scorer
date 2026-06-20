"""
MixUp Data Augmentation
Reference: Zhang et al., "MixUp: Beyond Empirical Risk Minimization",
           ICLR 2018.

Linearly interpolates pairs of training examples and their labels:
    x_mix = λ * x_i + (1 - λ) * x_j
    y_mix = λ * y_i + (1 - λ) * y_j
where λ ~ Beta(α, α).
"""

import numpy as np
import torch


def mixup_data(
    images: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.4,
) -> tuple:
    """
    Apply MixUp to a batch.

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
    if alpha > 0:
        lam = float(np.random.beta(alpha, alpha))
    else:
        lam = 1.0

    batch_size = images.size(0)
    perm = torch.randperm(batch_size, device=images.device)

    mixed = lam * images + (1.0 - lam) * images[perm]
    return mixed, targets, targets[perm], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute the MixUp loss."""
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)
