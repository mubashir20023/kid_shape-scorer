"""
make_fold_accuracy.py — CV-fold accuracy figure, built directly from
cv_summary.json (which has NO per-epoch history — only one best_val_acc
and one test_acc per fold). Replaces the old "curve" figure, which can
never be recovered because that per-epoch data was never logged during
training.

Usage:
  python make_fold_accuracy.py
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from config import SHAPES

OUT  = os.path.join(ROOT, "report_overleaf", "figures")
JSON = os.path.join(ROOT, "out", "cv_summary.json")   # <-- lives in out/, not project root
os.makedirs(OUT, exist_ok=True)

DISP = {"Diagonal": "Diamond", "Overlapped circle": "Overlapped Circles",
        "Overlapped pencils": "Overlapped Pencils"}
def disp(n): return DISP.get(n, n)

def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=160)
    plt.close(fig)
    print("wrote", name)

data = json.load(open(JSON))

# best model per shape, by mean acc_mean (matches Table 1's bolding)
best = {}
for r in data:
    s = r["shape"]
    if s not in best or r["acc_mean"] > best[s]["acc_mean"]:
        best[s] = r

fig, axes = plt.subplots(2, 4, figsize=(13, 6.2))
for ax, shape in zip(axes.ravel(), SHAPES):
    r = best.get(shape)
    if r is None or not r.get("fold_results"):
        ax.text(0.5, 0.5, "no fold data", ha="center", va="center",
                fontsize=8, color="gray", transform=ax.transAxes)
        ax.set_title(f"{disp(shape)}", fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([])
        continue

    folds = r["fold_results"]
    xs = [f["fold"] + 1 for f in folds]
    val = [f["best_val_acc"] * 100 for f in folds]
    test = [f["test_acc"] * 100 for f in folds]

    ax.plot(xs, val, marker="o", ms=5, label="Best Val", linestyle="--")
    ax.plot(xs, test, marker="s", ms=5, label="Test")
    ax.axhline(r["acc_mean"] * 100, color="gray", linewidth=1,
               linestyle=":", label="Mean Test")
    ax.set_title(f"{disp(shape)} ({r['model']})", fontsize=9.5)
    ax.set_xlabel("Fold", fontsize=8)
    ax.set_ylabel("Acc (%)", fontsize=8)
    ax.set_xticks(xs)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=6)

fig.suptitle("Per-Fold Accuracy (best model per shape, 5-fold CV)",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
save(fig, "fold_accuracy")