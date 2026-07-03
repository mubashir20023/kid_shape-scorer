"""
train_cv.py — Stratified k-fold cross-validation training for BOT-2
              hand-drawn shape scoring.

This mirrors train.py's model/optimiser/augmentation setup exactly, but
replaces the single 70/20/10 split with N_FOLDS-fold cross-validation:
for each fold, the held-out fold is the test set for that iteration, and
an inner train/val split (within the fold's training portion) is used
for checkpoint selection, exactly as in the single-split protocol.

Singleton classes are kept in every fold's training set and never scored
(same behaviour as train.py / build_dataloaders).

Usage:
  python train_cv.py                 # all shapes, both models
  python train_cv.py --shape Circle  # one shape
  python train_cv.py --folds 5
"""

import argparse
import copy
import ctypes
import gc
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, cohen_kappa_score, mean_absolute_error
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import (RESULTS_DIR, SHAPES, BATCH_SIZE, SEED, WEIGHT_DECAY,
                    LABEL_SMOOTHING, CUTMIX_PROB, MIXUP_PROB,
                    CUTMIX_ALPHA, MIXUP_ALPHA, N_FOLDS, CV_INNER_VAL_SPLIT)
from dataset import build_cv_dataloaders
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
CV_SUMMARY_PATH = os.path.join(RESULTS_DIR, "cv_summary.json")
PROGRESS_DIR    = os.path.join(RESULTS_DIR, "cv_progress")
os.makedirs(PROGRESS_DIR, exist_ok=True)


def _progress_path(shape: str, tag: str) -> str:
    safe_shape = shape.replace(" ", "_")
    safe_tag   = tag.replace("-", "").replace(" ", "_")
    return os.path.join(PROGRESS_DIR, f"{safe_shape}_{safe_tag}.json")


def _load_progress(shape: str, tag: str) -> dict:
    """Returns {fold_idx: fold_result_dict} for folds already completed."""
    path = _progress_path(shape, tag)
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        return {f["fold"]: f for f in data}
    except Exception:
        return {}


def _save_progress(shape: str, tag: str, fold_results: list):
    path = _progress_path(shape, tag)
    with open(path, "w") as f:
        json.dump(fold_results, f, indent=2)


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


def _adjacent_acc(labels, preds):
    return float(np.mean(np.abs(np.array(labels) - np.array(preds)) <= 1))


def train_one_fold(shape, model_type, fold_info, nc, fold_idx, n_folds):
    """Train on a single CV fold; returns metrics + preds/labels for that fold."""
    tr = fold_info["train_loader"]
    va = fold_info["val_loader"]
    te = fold_info["test_loader"]

    train_labels = [l for _, l in tr.dataset.samples]
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

    best_val_acc   = 0.0
    best_state_dict = None

    print(f"    -- fold {fold_idx + 1}/{n_folds} "
          f"(train={fold_info['n_train']} val={fold_info['n_val']} test={fold_info['n_test']}) --",
          flush=True)

    for ep in range(1, NUM_EPOCHS + 1):
        tl, ta = train_one_epoch(model, tr, opt, crit, use_mix)
        vl, vacc, _, _ = evaluate(model, va, crit)
        sched.step()
        if vacc > best_val_acc:
            best_val_acc = vacc
            best_state_dict = copy.deepcopy(model.state_dict())
        print(f"      ep {ep:2d}/{NUM_EPOCHS}  train {ta*100:5.1f}%  val {vacc*100:5.1f}%",
              flush=True)

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    _, tacc, P, L = evaluate(model, te, crit)
    mae = mean_absolute_error(L, P) if len(L) else float("nan")
    adj = _adjacent_acc(L, P) if len(L) else float("nan")
    try:
        qwk = cohen_kappa_score(L, P, weights="quadratic") if len(L) else float("nan")
    except Exception:
        qwk = float("nan")

    print(f"      fold {fold_idx + 1} TEST acc {tacc*100:.1f}%  "
          f"MAE {mae:.3f}  QWK {qwk:.3f}", flush=True)

    result = {
        "fold": fold_idx,
        "best_val_acc": best_val_acc,
        "test_acc": tacc,
        "mae": mae,
        "adj_acc": adj,
        "qwk": qwk,
        "preds": P.tolist(),
        "labels": L.tolist(),
    }

    # Explicit cleanup — each fold instantiates a brand-new ViT-S/16 or
    # MobileViT-S model (+ optimiser + scheduler). Without dropping these
    # references and forcing a collection, CPU memory accumulates across
    # folds (and across shapes, over an 8-shape x 2-model x 5-fold run)
    # rather than being returned promptly, which is what caused the
    # "not enough memory" crash on later folds.
    del model, opt, sched, params, best_state_dict
    gc.collect()

    return result


def train_shape_cv(shape, model_type, n_folds):
    set_seed(SEED)
    dm = DiffuseMix(alpha=0.5, use_diffusion=False) if _DM_OK else None

    tag = "ViT-S" if model_type == "vit_s" else "MobileViT-S"

    # Building fold_loaders just re-derives the file/label splits (cheap —
    # no model or images are loaded into memory yet), so it's safe to redo
    # this every time we resume without wasting significant time.
    fold_loaders, nc, cw, remap = build_cv_dataloaders(
        shape, n_folds=n_folds, batch_size=BATCH_SIZE, diffusemix=dm,
        seed=SEED, inner_val_split=CV_INNER_VAL_SPLIT,
    )

    print(f"\n=== CV | {tag} | {shape} | {nc} classes | {len(fold_loaders)} folds ===",
          flush=True)

    completed = _load_progress(shape, tag)
    if completed:
        print(f"  resuming: {len(completed)}/{len(fold_loaders)} fold(s) "
              f"already completed for {shape} / {tag}", flush=True)

    fold_results = []
    for fold_idx, fold_info in enumerate(fold_loaders):
        if fold_idx in completed:
            print(f"    -- fold {fold_idx + 1}/{len(fold_loaders)} "
                  f"skipped (already completed) --", flush=True)
            fold_results.append(completed[fold_idx])
            continue

        result = train_one_fold(shape, model_type, fold_info, nc,
                                fold_idx, len(fold_loaders))
        fold_results.append(result)
        # Persist immediately so a crash only ever costs the current fold,
        # not the folds already finished for this shape/model.
        _save_progress(shape, tag, fold_results)

    # Aggregate across folds
    accs = [f["test_acc"] for f in fold_results]
    maes = [f["mae"]      for f in fold_results]
    adjs = [f["adj_acc"]  for f in fold_results]
    qwks = [f["qwk"]      for f in fold_results if not np.isnan(f["qwk"])]

    all_preds  = [p for f in fold_results for p in f["preds"]]
    all_labels = [l for f in fold_results for l in f["labels"]]

    report = classification_report(
        all_labels, all_preds, output_dict=True, zero_division=0
    ) if all_labels else {}

    summary = {
        "shape": shape,
        "model": tag,
        "n_folds": len(fold_loaders),
        "acc_mean": float(np.mean(accs)),  "acc_std": float(np.std(accs)),
        "mae_mean": float(np.mean(maes)),  "mae_std": float(np.std(maes)),
        "adj_mean": float(np.mean(adjs)),  "adj_std": float(np.std(adjs)),
        "qwk_mean": float(np.mean(qwks)) if qwks else float("nan"),
        "qwk_std":  float(np.std(qwks))  if qwks else float("nan"),
        "fold_results": fold_results,
        "pooled_classification_report": report,
        "label_remap": {str(k): int(v) for k, v in remap.items()},
    }

    print(f"  CV summary [{shape} / {tag}]: "
          f"acc {summary['acc_mean']*100:.1f}±{summary['acc_std']*100:.1f}%  "
          f"MAE {summary['mae_mean']:.3f}±{summary['mae_std']:.3f}  "
          f"QWK {summary['qwk_mean']:.3f}±{summary['qwk_std']:.3f}", flush=True)

    # This shape/model is now fully recorded in cv_summary.json by the
    # caller — clean up the now-redundant per-fold progress file.
    progress_path = _progress_path(shape, tag)
    if os.path.exists(progress_path):
        os.remove(progress_path)

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="all")
    ap.add_argument("--model", default="both", choices=["vit_s", "mobile", "both"])
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    args = ap.parse_args()

    keep_awake()
    shapes = list(SHAPES) if args.shape == "all" else [args.shape]
    mts    = ["vit_s", "mobile"] if args.model == "both" else [args.model]

    res = []
    if os.path.exists(CV_SUMMARY_PATH):
        try: res = json.load(open(CV_SUMMARY_PATH))
        except Exception: res = []
    done = {(r["shape"], r["model"]) for r in res}

    print(f"Device: {DEVICE} | {args.folds}-fold CV | "
          f"{len(done)} combo(s) already done", flush=True)

    for s in shapes:
        for mt in mts:
            tag = "ViT-S" if mt == "vit_s" else "MobileViT-S"
            if (s, tag) in done:
                print(f"skip {s} {tag} (already done)", flush=True)
                continue
            res.append(train_shape_cv(s, mt, args.folds))
            json.dump(res, open(CV_SUMMARY_PATH, "w"), indent=2)
            print(f"  saved {len(res)}/16 -> {CV_SUMMARY_PATH}", flush=True)

    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()