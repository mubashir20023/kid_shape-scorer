"""
mobilevit.py — MobileViT-S backbone for BOT-2 shape scoring.
"""

import torch
import torch.nn as nn

try:
    import timm
    _TIMM_OK = True
except ImportError:
    _TIMM_OK = False


class MobileViTHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(in_features, num_classes)
        nn.init.trunc_normal_(self.fc.weight, std=0.02)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x.mean(dim=(2, 3))
        x = self.norm(x)
        x = self.drop(x)
        return self.fc(x)


def build_mobilevit_s(
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    if not _TIMM_OK:
        raise ImportError("timm is required.  pip install timm")

    model = timm.create_model(
        "mobilevit_s",
        pretrained=pretrained,
        num_classes=0,
    )

    with torch.no_grad():
        dummy   = torch.zeros(1, 3, 256, 256)
        feat    = model(dummy)
        if feat.ndim == 4:
            feat = feat.mean(dim=(2, 3))
        feat_dim = feat.shape[-1]

    model.head = MobileViTHead(feat_dim, num_classes, dropout)
    return model


class MobileViTWrapper(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True,
                 dropout: float = 0.2):
        super().__init__()
        self.backbone = build_mobilevit_s(num_classes, pretrained, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            param.requires_grad = "head" in name

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_param_groups(
        self,
        head_lr: float,
        backbone_lr: float,
        weight_decay: float = 1e-2,
    ) -> list:
        no_decay = {"bias", "norm"}

        def _no_wd(name):
            return any(nd in name for nd in no_decay)

        head_params     = [(n, p) for n, p in self.backbone.named_parameters()
                           if "head" in n]
        backbone_params = [(n, p) for n, p in self.backbone.named_parameters()
                           if "head" not in n]

        return [
            {"params": [p for _, p in head_params if not _no_wd(_)],
             "lr": head_lr, "weight_decay": weight_decay},
            {"params": [p for n, p in head_params if _no_wd(n)],
             "lr": head_lr, "weight_decay": 0.0},
            {"params": [p for n, p in backbone_params if not _no_wd(n)],
             "lr": backbone_lr, "weight_decay": weight_decay},
            {"params": [p for n, p in backbone_params if _no_wd(n)],
             "lr": backbone_lr, "weight_decay": 0.0},
        ]