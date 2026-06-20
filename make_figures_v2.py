"""
make_figures_v2.py — polished figures for the report, written into
overleaf/figures/.

Figures:
  sample_shapes.pdf         one real drawing per shape
  class_distribution.pdf    per-shape score distribution
  augmentation_pipeline.pdf Original / Geometric / MixUp / CutMix / DiffuseMix
  confusion_matrices.pdf    montage from a results JSON (best model per shape)

Usage:
  python make_figures_v2.py --json results/training_summary.json
"""

import argparse
import os
import sys
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm as mpl_cm
from PIL import Image
import torch
from torchvision import transforms

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from config import SHAPES                                   # noqa: E402
from dataset import load_shape_samples, get_val_transforms  # noqa: E402
from augmentation.diffusemix import DiffuseMix              # noqa: E402
from augmentation.mixup import mixup_data                   # noqa: E402

OUT = os.path.join(ROOT, "overleaf", "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

DISP = {"Diagonal": "Diamond", "Overlapped circle": "Overlapped Circles",
        "Overlapped pencils": "Overlapped Pencils"}
def disp(n): return DISP.get(n, n)

BLUE = "#3B6FB0"


def _save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=160)
    plt.close(fig)
    print("wrote", name)


def fig_sample_shapes():
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.4))
    for ax, shape in zip(axes.ravel(), SHAPES):
        s = load_shape_samples(shape)
        s.sort(key=lambda x: x[1])
        path, score = s[len(s) // 2]
        ax.imshow(Image.open(path).convert("RGB"))
        ax.set_title(f"{disp(shape)}  (score {score})", fontsize=10)
        ax.axis("off")
    fig.suptitle("Representative BOT-2 Drawings", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "sample_shapes")


def fig_class_distribution():
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.2))
    fig.subplots_adjust(hspace=0.55, wspace=0.35)
    for ax, shape in zip(axes.ravel(), SHAPES):
        labels = [s[1] for s in load_shape_samples(shape)]
        dist = Counter(labels)
        mx = SHAPES[shape]["max_score"]
        xs = list(range(mx + 1))
        ys = [dist.get(x, 0) for x in xs]
        norm = [(y / max(ys)) if max(ys) else 0 for y in ys]
        colors = [mpl_cm.Blues(0.35 + 0.6 * n) for n in norm]
        bars = ax.bar(xs, ys, color=colors, edgecolor="white", linewidth=0.6)
        ax.set_ylim(0, max(ys) * 1.22)
        ax.set_title(f"{disp(shape)}  (n={sum(ys)})", fontsize=10)
        ax.set_xticks(xs)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("Score", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        for b, y in zip(bars, ys):
            if y > 0:
                ax.text(b.get_x() + b.get_width() / 2, y + max(ys) * 0.02,
                        str(y), ha="center", va="bottom", fontsize=7.5)
    fig.suptitle("Expert-Score Distribution Across BOT-2 Shape Categories",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "class_distribution")


def _to_pil(t):
    arr = (t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def fig_augmentation_pipeline(seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    a = load_shape_samples("Circle"); a.sort(key=lambda x: x[1])
    b = load_shape_samples("Square"); b.sort(key=lambda x: x[1])
    img_a = Image.open(a[len(a) // 2][0]).convert("RGB")
    img_b = Image.open(b[len(b) // 2][0]).convert("RGB")
    size = 224
    rs = transforms.Resize((size, size))
    img_a, img_b = rs(img_a), rs(img_b)

    geom = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.3, 0.3, 0.2, 0.1),
        transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.85, 1.15))])
    geom_img = geom(img_a)

    dm = DiffuseMix(alpha=0.5, use_diffusion=False)
    dm_img = dm(img_a)

    to_t = transforms.ToTensor()
    batch = torch.stack([to_t(img_a), to_t(img_b)])
    mu, *_ = mixup_data(batch.clone(), torch.tensor([0, 1]), alpha=0.4)
    cm = batch[0].clone(); m = int(size * 0.16)
    cm[:, m:size - m, m:size - m] = batch[1, :, m:size - m, m:size - m]

    panels = [(img_a, "Original"), (geom_img, "Geometric"),
              (_to_pil(mu[0]), "MixUp"), (_to_pil(cm), "CutMix"),
              (dm_img, "DiffuseMix")]
    fig, axes = plt.subplots(1, 5, figsize=(13, 2.9))
    for ax, (im, t) in zip(axes, panels):
        ax.imshow(im); ax.set_title(t, fontsize=11); ax.axis("off")
    fig.tight_layout()
    _save(fig, "augmentation_pipeline")


def fig_confusion(json_path):
    import json
    from sklearn.metrics import confusion_matrix
    if not os.path.exists(json_path):
        print("no results json -> skipping confusion montage"); return
    data = json.load(open(json_path))
    best = {}
    for r in data:
        s = r["shape"]
        if s not in best or r["test_acc"] > best[s]["test_acc"]:
            best[s] = r
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.4))
    for ax, shape in zip(axes.ravel(), SHAPES):
        r = best.get(shape)
        if r is None:
            ax.axis("off"); continue
        preds = np.array(r["preds"]); labs = np.array(r["labels"])
        remap = r.get("label_remap", {})
        inv = {}
        for orig, mapped in remap.items():
            inv.setdefault(int(mapped), []).append(int(orig))
        K = max(int(max(remap.values())) + 1 if remap else 0,
                int(labs.max()) + 1 if len(labs) else 1)
        ticks = ["/".join(str(x) for x in inv.get(i, [i])) for i in range(K)]
        cmtx = confusion_matrix(labs, preds, labels=list(range(K)))
        im = ax.imshow(cmtx, cmap="Blues")
        ax.set_title(f"{disp(shape)} ({r['model']})", fontsize=9.5)
        ax.set_xticks(range(K)); ax.set_yticks(range(K))
        ax.set_xticklabels(ticks, fontsize=7); ax.set_yticklabels(ticks, fontsize=7)
        ax.set_xlabel("Predicted", fontsize=8); ax.set_ylabel("True", fontsize=8)
        thr = cmtx.max() / 2 if cmtx.max() else 0
        for i in range(K):
            for j in range(K):
                if cmtx[i, j]:
                    ax.text(j, i, cmtx[i, j], ha="center", va="center",
                            fontsize=7, color="white" if cmtx[i, j] > thr else "black")
    fig.suptitle("Confusion Matrices (best model per shape)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "confusion_matrices")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(ROOT, "results",
                                                   "training_summary.json"))
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    OUT = args.out
    os.makedirs(OUT, exist_ok=True)
    fig_sample_shapes()
    fig_class_distribution()
    fig_augmentation_pipeline()
    fig_confusion(args.json)
    print("All figures ->", OUT)