"""
train.py — Linear-probe training for BOT-2 hand-drawn shape scoring.

Usage:
  python train.py                 # all shapes, both models
  python train.py --shape Circle  # one shape
"""

import argparse
import ctypes
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from config import (RESULTS_DIR, SHAPES, BATCH_SIZE, SEED, WEIGHT_DECAY,
                    LABEL_SMOOTHING, CUTMIX_PROB, MIXUP_PROB,
                    CUTMIX_ALPHA, MIXUP_ALPHA)
from dataset import build_dataloaders
from models.vit_small import ViTSmallWrapper
from models.mobilevit import MobileViTWrapper

try:
    from augmentation.diffusemix import DiffuseMix
    _DM_OK = True
except ImportError:
    _DM_OK = False
from augmentation.cutmix import cutmix_data, cutmix_criterion
from augmentation.mixup import mixup_data, mixup_criterion

NUM_EPOCHS   = 10
DEVICE       = torch.device("cpu")
SUMMARY_PATH = os.path.join(RESULTS_DIR, "training_summary.json")


def keep_awake():
    if sys.platform == "win32":
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        except Exception:
            pass


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def mild_weights(labels, num_classes):
    c = np.bincount(labels, minlength=num_classes).astype(np.float32)
    present = c > 0
    safe = np.where(c == 0, 1.0, c)
    w = c.sum() / (num_classes * safe)
    w = np.clip(w, 0.5, 3.0)
    w = w / w[present].mean()
    w[~present] = 1.0
    return torch.tensor(w, dtype=torch.float32)


def train_one_epoch(model, loader, opt, crit, use_mix):
    model.train()
    tot = corr = 0
    tloss = 0.0
    for img, lab in loader:
        img, lab = img.to(DEVICE), lab.to(DEVICE)
        if use_mix:
            r = random.random()
            if r < CUTMIX_PROB:
                img, ya, yb, lam = cutmix_data(img, lab, CUTMIX_ALPHA)
                out = model(img); loss = cutmix_criterion(crit, out, ya, yb, lam)
            elif r < CUTMIX_PROB + MIXUP_PROB:
                img, ya, yb, lam = mixup_data(img, lab, MIXUP_ALPHA)
                out = model(img); loss = mixup_criterion(crit, out, ya, yb, lam)
            else:
                out = model(img); loss = crit(out, lab)
        else:
            out = model(img); loss = crit(out, lab)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tloss += loss.item() * img.size(0)
        corr  += (out.argmax(1) == lab).sum().item()
        tot   += img.size(0)
    return tloss / tot, corr / tot


@torch.no_grad()
def evaluate(model, loader, crit):
    model.eval()
    tot = corr = 0
    tloss = 0.0
    P, L = [], []
    for img, lab in loader:
        img, lab = img.to(DEVICE), lab.to(DEVICE)
        out = model(img); loss = crit(out, lab)
        tloss += loss.item() * lab.size(0)
        pr = out.argmax(1)
        corr += (pr == lab).sum().item(); tot += lab.size(0)
        P += pr.cpu().tolist(); L += lab.cpu().tolist()
    return tloss / tot, corr / tot, np.array(P), np.array(L)


def train_shape(shape, model_type):
    set_seed(SEED)
    dm = DiffuseMix(alpha=0.5, use_diffusion=False) if _DM_OK else None
    tr, va, te, nc, cw, remap = build_dataloaders(
        shape, batch_size=BATCH_SIZE, diffusemix=dm)

    train_ds = tr.dataset
    tr = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                    num_workers=0, drop_last=len(train_ds) > BATCH_SIZE)
    train_labels = [l for _, l in train_ds.samples]
    mild_cw = mild_weights(train_labels, nc).to(DEVICE)

    tag = "ViT-S" if model_type == "vit_s" else "MobileViT-S"
    maj_frac = max(np.bincount(train_labels)) / len(train_labels)
    extreme  = maj_frac > 0.85

    if model_type == "vit_s":
        model   = ViTSmallWrapper(nc, pretrained=True)
        lr      = 1e-4
        crit    = nn.CrossEntropyLoss(
            weight=None if extreme else mild_cw, label_smoothing=LABEL_SMOOTHING)
        use_mix = True
    else:
        model   = MobileViTWrapper(nc, pretrained=True)
        lr      = 3e-3
        crit    = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        use_mix = False

    model = model.to(DEVICE)
    model.freeze_backbone()
    params = [p for p in model.parameters() if p.requires_grad]
    opt    = AdamW(params, lr=lr, weight_decay=WEIGHT_DECAY)
    sched  = CosineAnnealingLR(opt, T_max=NUM_EPOCHS)

    best = 0.0
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    save = os.path.join(RESULTS_DIR, f"{shape}_{model_type}_clean.pt")

    print(f"\n=== {tag} | {shape} | {nc} classes | lr={lr} mix={use_mix} ===",
          flush=True)
    for ep in range(1, NUM_EPOCHS + 1):
        tl, ta = train_one_epoch(model, tr, opt, crit, use_mix)
        vl, vacc, _, _ = evaluate(model, va, crit)
        sched.step()
        hist["train_loss"].append(tl); hist["val_loss"].append(vl)
        hist["train_acc"].append(ta);  hist["val_acc"].append(vacc)
        if vacc > best:
            best = vacc
            torch.save(model.state_dict(), save)
        print(f"  ep {ep:2d}/{NUM_EPOCHS}  train {ta*100:5.1f}%  val {vacc*100:5.1f}%",
              flush=True)

    model.load_state_dict(torch.load(save, map_location=DEVICE))
    _, tacc, P, L = evaluate(model, te, crit)
    per = {int(c): float((P[L == c] == c).mean()) for c in sorted(set(L.tolist()))}
    print(f"  TEST acc {tacc*100:.1f}%", flush=True)

    return {
        "shape": shape, "model": tag, "best_val_acc": best, "test_acc": tacc,
        "per_class_acc": per, "history": hist,
        "preds": P.tolist(), "labels": L.tolist(),
        "label_remap": {str(k): int(v) for k, v in remap.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="all")
    ap.add_argument("--model", default="both", choices=["vit_s", "mobile", "both"])
    args = ap.parse_args()

    keep_awake()
    shapes = list(SHAPES) if args.shape == "all" else [args.shape]
    mts    = ["vit_s", "mobile"] if args.model == "both" else [args.model]

    res = []
    if os.path.exists(SUMMARY_PATH):
        try: res = json.load(open(SUMMARY_PATH))
        except Exception: res = []
    done = {(r["shape"], r["model"]) for r in res}

    print(f"Device: {DEVICE} | linear-probe | {len(done)} combo(s) already done",
          flush=True)
    for s in shapes:
        for mt in mts:
            tag = "ViT-S" if mt == "vit_s" else "MobileViT-S"
            if (s, tag) in done:
                print(f"skip {s} {tag} (already done)", flush=True)
                continue
            res.append(train_shape(s, mt))
            json.dump(res, open(SUMMARY_PATH, "w"), indent=2)
            print(f"  saved {len(res)}/16 -> {SUMMARY_PATH}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()