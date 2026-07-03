"""
Drop-in replacement for the val_accuracy block in your figure script.
Adds diagnostics so you can see *why* a panel is empty, and fixes the
two most common causes:
  1. history missing / val_acc empty  -> label the panel instead of
     silently leaving a blank box.
  2. train_acc/val_acc already stored as percentages (0-100) instead
     of fractions (0-1) -> the old code's `*100` pushed points off
     the y-limit (0, 100), so the line was drawn but invisible.
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from config import SHAPES

OUT  = os.path.join(ROOT, "report_overleaf", "figures")
JSON = os.path.join(ROOT, "out", "training_summary.json")
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
best = {}
for r in data:
    s = r["shape"]
    if s not in best or r["test_acc"] > best[s]["test_acc"]:
        best[s] = r

# ---- one-time diagnostic: what does `history` actually look like? ----
print("\n--- history diagnostic ---")
for shape, r in best.items():
    h = r.get("history")
    if not h:
        print(f"{shape:20s} model={r['model']:12s} -> NO 'history' key at all")
        continue
    va = h.get("val_acc") or h.get("val_accuracy")
    ta = h.get("train_acc") or h.get("train_accuracy")
    if not va:
        print(f"{shape:20s} model={r['model']:12s} -> history present but "
              f"val_acc/val_accuracy missing or empty. keys={list(h.keys())}")
        continue
    print(f"{shape:20s} model={r['model']:12s} -> {len(va)} epochs, "
          f"val range=({min(va):.4f}, {max(va):.4f})")
print("--- end diagnostic ---\n")


def _accuracy_series(h, *keys):
    """Return the first matching key's values, auto-scaled to 0-100."""
    vals = None
    for k in keys:
        if h.get(k):
            vals = h[k]
            break
    if not vals:
        return None
    # If values already look like percentages, don't multiply again.
    return [v * 100 if v <= 1.0 else v for v in vals]


fig, axes = plt.subplots(2, 4, figsize=(13, 6.2))
for ax, shape in zip(axes.ravel(), SHAPES):
    r = best.get(shape)
    h = r.get("history") if r else None

    train_series = _accuracy_series(h, "train_acc", "train_accuracy") if h else None
    val_series   = _accuracy_series(h, "val_acc", "val_accuracy") if h else None

    if not val_series:
        # Show *why* it's empty instead of leaving a mystery blank box.
        ax.text(0.5, 0.5, "no per-epoch history\nlogged for this run",
                ha="center", va="center", fontsize=8, color="gray",
                transform=ax.transAxes)
        ax.set_title(f"{disp(shape)} ({r['model'] if r else '?'})", fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([])
        continue

    ep = range(1, len(val_series) + 1)
    if train_series:
        ax.plot(ep, train_series, marker="o", ms=3, label="Train")
    ax.plot(ep, val_series, marker="s", ms=3, label="Val")
    ax.set_title(f"{disp(shape)} ({r['model']})", fontsize=9.5)
    ax.set_xlabel("Epoch", fontsize=8)
    ax.set_ylabel("Acc (%)", fontsize=8)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7)

fig.suptitle("Validation Accuracy per Shape (best model)", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
save(fig, "val_accuracy")