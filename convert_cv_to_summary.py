import json
import os

cv_file = "out/cv_summary.json"
out_file = "out/training_summary.json"

data = json.load(open(cv_file))

summary = []

for item in data:

    preds = []
    labels = []

    for fold in item["fold_results"]:
        preds.extend(fold["preds"])
        labels.extend(fold["labels"])

    summary.append({
        "shape": item["shape"],
        "model": item["model"],

        "test_acc": item["acc_mean"],
        "qwk": item["qwk_mean"],
        "mae": item["mae_mean"],

        "preds": preds,
        "labels": labels,

        "label_remap": item.get("label_remap", {}),

        "history": {
            "train_loss": [0],
            "val_loss": [0],
            "train_acc": [0],
            "val_acc": [0]
        }
    })


json.dump(summary, open(out_file, "w"), indent=2)

print("DONE:", out_file)