"""Generate Plan-A result figures (confusion matrices + validation curves) into
report_overleaf/figures, and print the baseline MAE for the paper."""
import json, os, sys
from collections import Counter
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, mean_absolute_error

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from config import SHAPES

OUT  = os.path.join(ROOT, "report_overleaf", "figures")
JSON = os.path.join(ROOT, "results", "training_summary_clean.json")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlesize": 10,
                     "axes.titleweight": "bold", "figure.dpi": 150, "savefig.bbox": "tight"})
DISP = {"Diagonal": "Diamond", "Overlapped circle": "Overlapped Circles",
        "Overlapped pencils": "Overlapped Pencils"}
def disp(n): return DISP.get(n, n)
def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf")); fig.savefig(os.path.join(OUT, name + ".png"), dpi=160)
    plt.close(fig); print("wrote", name)

data = json.load(open(JSON))
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
    for orig, mp in remap.items():
        inv.setdefault(int(mp), int(orig))
    K = (max(int(x) for x in remap.values()) + 1) if remap else int(max(labs.max(), preds.max())) + 1
    ticks = [inv.get(i, i) for i in range(K)]
    cm = confusion_matrix(labs, preds, labels=list(range(K)))
    ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{disp(shape)} ({r['model']})", fontsize=9.5)
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels(ticks, fontsize=7); ax.set_yticklabels(ticks, fontsize=7)
    ax.set_xlabel("Predicted", fontsize=8); ax.set_ylabel("True", fontsize=8)
    thr = cm.max() / 2 if cm.max() else 0
    for i in range(K):
        for j in range(K):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > thr else "black")
fig.suptitle("Confusion Matrices (best model per shape)", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95]); save(fig, "confusion_matrices")

fig, axes = plt.subplots(2, 4, figsize=(13, 6.2))
for ax, shape in zip(axes.ravel(), SHAPES):
    r = best.get(shape)
    h = r["history"] if r else None
    if not h or not h.get("val_acc"):
        ax.axis("off"); continue
    ep = range(1, len(h["val_acc"]) + 1)
    ax.plot(ep, [a*100 for a in h["train_acc"]], marker="o", ms=3, label="Train")
    ax.plot(ep, [a*100 for a in h["val_acc"]],   marker="s", ms=3, label="Val")
    ax.set_title(f"{disp(shape)} ({r['model']})", fontsize=9.5)
    ax.set_xlabel("Epoch", fontsize=8); ax.set_ylabel("Acc (%)", fontsize=8)
    ax.set_ylim(0, 100); ax.legend(fontsize=7)
fig.suptitle("Validation Accuracy per Shape (best model)", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95]); save(fig, "val_accuracy")

maes = []
for r in data:
    if r["model"] != "ViT-S":
        continue
    labs = r["labels"]
    if not labs: continue
    maj = Counter(labs).most_common(1)[0][0]
    maes.append(mean_absolute_error(labs, [maj] * len(labs)))
print(f"\nBaseline (majority predictor) mean MAE = {np.mean(maes):.2f}")