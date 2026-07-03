"""
evaluate.py — Generates tables, confusion matrices, and result plots
              for the BOT-2 shape scoring research paper.

Usage
-----
  python evaluate.py                    # loads results/training_summary.json
  python evaluate.py --json my_run.json
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    confusion_matrix, cohen_kappa_score, mean_absolute_error,
    classification_report,
)

from config import RESULTS_DIR, SHAPES


def load_results(json_path: str) -> list:
    with open(json_path) as f:
        return json.load(f)


def _save(fig, name: str, save_dir: str):
    base = os.path.join(save_dir, name)
    fig.savefig(base + ".pdf", dpi=150, bbox_inches="tight")
    fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(labels, preds, shape_name: str, model_tag: str,
                          num_classes: int, label_remap: dict, save_dir: str):
    if label_remap:
        inv = {}
        for orig, mapped in label_remap.items():
            inv.setdefault(mapped, int(orig))
        tick_labels = [inv.get(i, i) for i in range(num_classes)]
    else:
        tick_labels = list(range(num_classes))

    cm  = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=range(num_classes), yticks=range(num_classes),
        xticklabels=tick_labels,   yticklabels=tick_labels,
        xlabel="Predicted score", ylabel="True score",
        title=f"{shape_name} — {model_tag}",
    )
    thresh = cm.max() / 2
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8)
    plt.tight_layout()
    fname = f"cm_{shape_name.replace(' ', '_')}_{model_tag.replace('-', '')}"
    _save(fig, fname, save_dir)


def plot_training_curves(history: dict, shape_name: str, model_tag: str,
                         save_dir: str):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"],   label="Val")
    ax1.set(xlabel="Epoch", ylabel="Loss",
            title=f"{shape_name} — {model_tag} | Loss")
    ax1.legend()

    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train")
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]],   label="Val")
    ax2.set(xlabel="Epoch", ylabel="Accuracy (%)",
            title=f"{shape_name} — {model_tag} | Accuracy")
    ax2.legend()

    plt.tight_layout()
    fname = f"curve_{shape_name.replace(' ', '_')}_{model_tag.replace('-', '')}"
    _save(fig, fname, save_dir)


def plot_metric_bars(results: list, metric: str, ylabel: str,
                     save_dir: str, higher_is_better: bool = True):
    vit_rows    = {r["shape"]: r for r in results if "ViT"    in r["model"]
                   and "Mobile" not in r["model"]}
    mobile_rows = {r["shape"]: r for r in results if "Mobile" in r["model"]}
    shapes      = list(SHAPES.keys())

    v_vals = [vit_rows.get(s,    {}).get(metric, float("nan")) for s in shapes]
    m_vals = [mobile_rows.get(s, {}).get(metric, float("nan")) for s in shapes]

    x   = np.arange(len(shapes))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w / 2, v_vals, w, label="ViT-S/16",    color="#4C72B0")
    ax.bar(x + w / 2, m_vals, w, label="MobileViT-S", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(shapes, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Per-shape {ylabel} comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, f"compare_{metric}", save_dir)


def _adjacent_acc(labels, preds) -> float:
    return float(np.mean(np.abs(np.array(labels) - np.array(preds)) <= 1))


def _compute_metrics(r: dict) -> dict:
    preds  = np.array(r["preds"])
    labels = np.array(r["labels"])
    mae    = mean_absolute_error(labels, preds)
    adj    = _adjacent_acc(labels, preds)
    try:
        qwk = cohen_kappa_score(labels, preds, weights="quadratic")
    except Exception:
        qwk = float("nan")
    return {
        "test_acc": r["test_acc"],
        "mae":      mae,
        "qwk":      qwk,
        "adj_acc":  adj,
    }


def print_latex_table(results: list):
    vit_rows    = {r["shape"]: r for r in results
                   if "ViT" in r["model"] and "Mobile" not in r["model"]}
    mobile_rows = {r["shape"]: r for r in results if "Mobile" in r["model"]}

    print("\n% --- Table: Per-shape test metrics ---")
    print(r"\begin{table}[t]\centering")
    print(r"\caption{Per-shape test metrics. Acc = exact accuracy, "
          r"Adj = adjacent accuracy ($\pm1$), QWK = quadratic weighted kappa.}")
    print(r"\label{tab:results}")
    print(r"\resizebox{\linewidth}{!}{\begin{tabular}{l c "
          r"ccc ccc c}")
    print(r"\hline")
    print(r" & & \multicolumn{3}{c}{\textbf{ViT-S/16}} "
          r"& \multicolumn{3}{c}{\textbf{MobileViT-S}} & \\")
    print(r"\textbf{Shape} & \textbf{Classes} "
          r"& Acc & Adj & QWK & Acc & Adj & QWK & \textbf{Best Acc} \\")
    print(r"\hline")

    for shape, info in SHAPES.items():
        nc = info["max_score"] + 1
        v  = _compute_metrics(vit_rows[shape])    if shape in vit_rows    else {}
        m  = _compute_metrics(mobile_rows[shape]) if shape in mobile_rows else {}

        def fmt(d, key, pct=True):
            val = d.get(key, float("nan"))
            return f"{val*100:.1f}" if pct else f"{val:.3f}"

        best_acc = max(v.get("test_acc", 0), m.get("test_acc", 0))
        best_tag = "V" if v.get("test_acc", 0) >= m.get("test_acc", 0) else "M"
        print(f"  {shape} & {nc} "
              f"& {fmt(v,'test_acc')} & {fmt(v,'adj_acc')} & {fmt(v,'qwk',False)} "
              f"& {fmt(m,'test_acc')} & {fmt(m,'adj_acc')} & {fmt(m,'qwk',False)} "
              f"& \\textbf{{{best_acc*100:.1f}}} ({best_tag}) \\\\")

    v_accs = [_compute_metrics(r)["test_acc"] for r in results
              if "ViT" in r["model"] and "Mobile" not in r["model"]]
    m_accs = [_compute_metrics(r)["test_acc"] for r in results
              if "Mobile" in r["model"]]
    print(r"  \hline")
    print(f"  \\textbf{{Mean}} & -- "
          f"& {np.mean(v_accs)*100:.1f} & -- & -- "
          f"& {np.mean(m_accs)*100:.1f} & -- & -- "
          f"& \\textbf{{{max(np.mean(v_accs), np.mean(m_accs))*100:.1f}}} \\\\")
    print(r"\hline\end{tabular}}\end{table}")


def extended_metrics(results: list):
    header = f"{'Shape':<22} {'Model':<14} {'Acc%':>6} {'Adj%':>6} {'MAE':>6} {'QWK':>6}"
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        m = _compute_metrics(r)
        print(f"  {r['shape']:<20} {r['model']:<14} "
              f"{m['test_acc']*100:>5.1f}% "
              f"{m['adj_acc']*100:>5.1f}% "
              f"{m['mae']:>6.3f} "
              f"{m['qwk']:>6.3f}")


def classification_reports(results: list, save_dir: str,
                           txt_name: str = "classification_reports.txt"):
    """
    Per-shape, per-model precision / recall / F1-score (in addition to the
    accuracy / QWK / MAE already reported by extended_metrics).

    Uses the original BOT-2 score (via label_remap) as the class label for
    readability, matching the confusion-matrix tick labels.

    Macro-F1 is printed alongside weighted-F1 because macro-F1 is the
    number that actually reflects performance on the rare, diagnostically
    important low-score classes; weighted-F1 (like raw accuracy) is
    dominated by the majority class and can look deceptively good.
    """
    lines  = []
    banner = "=" * 70

    for r in results:
        shape       = r["shape"]
        model_tag   = r["model"]
        label_remap = {int(k): v for k, v in r.get("label_remap", {}).items()}
        labels      = r["labels"]
        preds       = r["preds"]

        inv = {}
        for orig, mapped in label_remap.items():
            inv.setdefault(mapped, int(orig))

        present      = sorted(set(labels) | set(preds))
        target_names = [str(inv.get(i, i)) for i in present]

        report_txt = classification_report(
            labels, preds, labels=present, target_names=target_names,
            zero_division=0,
        )
        report_dict = classification_report(
            labels, preds, labels=present, target_names=target_names,
            output_dict=True, zero_division=0,
        )
        macro_f1    = report_dict["macro avg"]["f1-score"]
        weighted_f1 = report_dict["weighted avg"]["f1-score"]

        block = (f"{banner}\n{shape} \u2014 {model_tag}  "
                 f"(macro-F1={macro_f1:.3f}, weighted-F1={weighted_f1:.3f})\n"
                 f"{banner}\n{report_txt}")
        print("\n" + block)
        lines.append(block)

        r["macro_f1"]    = macro_f1
        r["weighted_f1"] = weighted_f1

    out_path = os.path.join(save_dir, txt_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nFull per-class classification reports saved to {out_path}")


def print_prf_summary_table(results: list):
    header = (f"{'Shape':<22} {'Model':<14} {'Acc%':>6} "
              f"{'MacroF1':>8} {'WtF1':>8} {'QWK':>6}")
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        print(f"  {r['shape']:<20} {r['model']:<14} "
              f"{r.get('test_acc', float('nan'))*100:>5.1f}% "
              f"{r.get('macro_f1', float('nan')):>8.3f} "
              f"{r.get('weighted_f1', float('nan')):>8.3f} "
              f"{r.get('qwk', float('nan')):>6.3f}")


def print_cv_summary_table(cv_json_path: str):
    if not os.path.exists(cv_json_path):
        return
    with open(cv_json_path) as f:
        cv_results = json.load(f)

    header = (f"{'Shape':<22} {'Model':<14} {'Folds':>5} "
              f"{'Acc% (mean±std)':>18} {'MAE (mean±std)':>16} "
              f"{'QWK (mean±std)':>16}")
    banner = "=" * len(header)
    print("\n" + banner)
    print("Cross-Validation Summary (mean \u00b1 std across folds)")
    print(banner)
    print(header)
    print("-" * len(header))
    for r in cv_results:
        acc_str = f"{r['acc_mean']*100:.1f}\u00b1{r['acc_std']*100:.1f}"
        mae_str = f"{r['mae_mean']:.3f}\u00b1{r['mae_std']:.3f}"
        qwk_str = f"{r['qwk_mean']:.3f}\u00b1{r['qwk_std']:.3f}"
        print(f"  {r['shape']:<20} {r['model']:<14} {r['n_folds']:>5} "
              f"{acc_str:>18} {mae_str:>16} {qwk_str:>16}")
    print(f"\n(loaded from {cv_json_path})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json",
                        default=os.path.join(RESULTS_DIR, "training_summary.json"))
    parser.add_argument("--cv_json",
                        default=os.path.join(RESULTS_DIR, "cv_summary.json"))
    args = parser.parse_args()

    results = load_results(args.json)

    for r in results:
        shape       = r["shape"]
        model_tag   = r["model"]
        nc          = SHAPES[shape]["max_score"] + 1
        label_remap = {int(k): v for k, v in r.get("label_remap", {}).items()}

        plot_confusion_matrix(r["labels"], r["preds"], shape, model_tag,
                              nc, label_remap, RESULTS_DIR)
        plot_training_curves(r["history"], shape, model_tag, RESULTS_DIR)

    for r in results:
        r.update(_compute_metrics(r))

    plot_metric_bars(results, "test_acc", "Exact accuracy (%)", RESULTS_DIR)
    plot_metric_bars(results, "qwk",      "Quadratic Weighted Kappa", RESULTS_DIR)
    plot_metric_bars(results, "mae",      "Mean Absolute Error",
                     RESULTS_DIR, higher_is_better=False)

    print_latex_table(results)
    extended_metrics(results)

    classification_reports(results, RESULTS_DIR)
    print_prf_summary_table(results)

    print_cv_summary_table(args.cv_json)

    print(f"\nAll plots saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()