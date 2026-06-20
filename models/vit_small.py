"""
vit_small.py — ViT-S/16 backbone for BOT-2 shape scoring.
"""

import torch
import torch.nn as nn

try:
    import timm
    _TIMM_OK = True
except ImportError:
    _TIMM_OK = False


class ViTSmallHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(in_features, num_classes)
        nn.init.trunc_normal_(self.fc.weight, std=0.02)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.drop(self.norm(x)))


def build_vit_small(
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.3,
    img_size: int = 224,
) -> nn.Module:
    if not _TIMM_OK:
        raise ImportError("timm is required.  pip install timm")

    model = timm.create_model(
        "vit_small_patch16_224",
        pretrained=pretrained,
        num_classes=0,
        img_size=img_size,
    )
    embed_dim  = model.embed_dim
    model.head = ViTSmallHead(embed_dim, num_classes, dropout)
    return model


def get_vit_param_groups(
    model: nn.Module,
    head_lr: float,
    backbone_lr: float,
    lr_decay: float = 0.75,
    weight_decay: float = 1e-2,
) -> list:
    no_decay = {"bias", "norm", "pos_embed", "cls_token", "dist_token"}

    def _no_wd(name: str) -> bool:
        return any(nd in name for nd in no_decay)

    num_blocks = len(model.blocks)
    groups     = []

    for n, p in model.head.named_parameters():
        groups.append({
            "params": [p],
            "lr": head_lr,
            "weight_decay": 0.0 if _no_wd(n) else weight_decay,
            "name": f"head.{n}",
        })

    for depth, block in enumerate(reversed(model.blocks)):
        layer_lr = backbone_lr * (lr_decay ** depth)
        for n, p in block.named_parameters():
            groups.append({
                "params": [p],
                "lr": layer_lr,
                "weight_decay": 0.0 if _no_wd(n) else weight_decay,
                "name": f"block{num_blocks - 1 - depth}.{n}",
            })

    stem_lr = backbone_lr * (lr_decay ** num_blocks)
    stem_params = {
        n: p
        for n, p in model.named_parameters()
        if not any(n.startswith(f"blocks.{i}") for i in range(num_blocks))
        and not n.startswith("head.")
    }
    for n, p in stem_params.items():
        groups.append({
            "params": [p],
            "lr": stem_lr,
            "weight_decay": 0.0 if _no_wd(n) else weight_decay,
            "name": f"stem.{n}",
        })

    return groups


class ViTSmallWrapper(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True,
                 dropout: float = 0.3):
        super().__init__()
        self.backbone = build_vit_small(num_classes, pretrained, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            param.requires_grad = "head" in name

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_param_groups(self, head_lr: float, backbone_lr: float,
                         lr_decay: float = 0.75, weight_decay: float = 1e-2):
        return get_vit_param_groups(
            self.backbone, head_lr, backbone_lr, lr_decay, weight_decay
        )