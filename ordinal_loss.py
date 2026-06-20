"""
ordinal_loss.py — CORN ordinal cross-entropy loss for BOT-2 shape scoring.

Reference: Shi et al., "Deep Neural Networks for Rank Consistent Ordinal
           Regression based on Conditional Probabilities", TMLR 2023.

Usage:
    from ordinal_loss import CornLoss, corn_predict
    criterion = CornLoss(num_classes, class_weights)
    loss = criterion(logits, labels)    # logits shape: (B, num_classes − 1)
    preds = corn_predict(logits)        # shape: (B,)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CornLoss(nn.Module):
    def __init__(self, num_classes: int, class_weights=None):
        super().__init__()
        self.num_classes   = num_classes
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        K   = self.num_classes
        B   = logits.size(0)
        dev = logits.device

        thresholds     = torch.arange(K - 1, device=dev).unsqueeze(0)
        binary_targets = (labels.unsqueeze(1) > thresholds).float()

        total_loss = torch.zeros(1, device=dev)
        total_n    = 0

        for j in range(K - 1):
            mask = labels >= j
            if mask.sum() == 0:
                continue
            logit_j  = logits[mask, j]
            target_j = binary_targets[mask, j]

            if self.class_weights is not None:
                w = self.class_weights[labels[mask]]
            else:
                w = torch.ones(mask.sum(), device=dev)

            loss_j     = F.binary_cross_entropy_with_logits(
                logit_j, target_j, weight=w, reduction="sum"
            )
            total_loss = total_loss + loss_j
            total_n   += mask.sum().item()

        return total_loss / max(total_n, 1)


def corn_predict(logits: torch.Tensor) -> torch.Tensor:
    cum_probs = torch.sigmoid(logits).cumprod(dim=1)
    ones  = torch.ones(cum_probs.size(0), 1, device=logits.device)
    zeros = torch.zeros(cum_probs.size(0), 1, device=logits.device)
    pmf   = torch.cat([ones, cum_probs, zeros], dim=1)
    pmf_class = pmf[:, :-1] - pmf[:, 1:]
    return pmf_class.argmax(dim=1)


def corn_logit_dim(num_classes: int) -> int:
    return num_classes - 1