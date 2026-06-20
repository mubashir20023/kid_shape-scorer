# BOT-2 Hand-Drawn Shape Scorer

A deep learning pipeline that automatically scores children's hand-drawn shapes from the **Bruininks-Oseretsky Test of Motor Proficiency (BOT-2)**. Two frozen ImageNet backbones — **ViT-S/16** and **MobileViT-S** — are trained as linear probes on top of expert-labelled drawings.

---

## Shapes

| Shape | Code | Max Score |
|---|---|---|
| Circle | C | 4 |
| Square | S | 5 |
| Triangle | T | 5 |
| Diagonal | D | 5 |
| Wave | W | 4 |
| Star | ST | 5 |
| Overlapped Circle | O | 6 |
| Overlapped Pencils | p | 6 |

---

## Project Structure

```
├── config.py                  # Central configuration
├── dataset.py                 # Data loading, augmentation, splitting
├── train.py                   # Training entry point
├── evaluate.py                # Metrics, confusion matrices, plots
├── make_figures_v2.py         # Publication-quality figures
├── make_results_figs.py       # Result figures for the report
├── load_data.py               # Raw image loader (CV2-based)
├── ordinal_loss.py            # CORN ordinal cross-entropy loss
├── models/
│   ├── vit_small.py           # ViT-S/16 wrapper
│   └── mobilevit.py           # MobileViT-S wrapper
├── augmentation/
│   ├── cutmix.py
│   ├── mixup.py
│   └── diffusemix.py
├── results/                   # Training outputs (auto-created)
└── report_overleaf/           # LaTeX report and figures
```

---

## Setup

**Requirements:** Python 3.9+

```bash
pip install -r requirements.txt
```

**Dataset:** Set `DATASET_PATH` in `config.py` to point to your local Shapes folder:

```python
DATASET_PATH = r"C:\path\to\Shapes"
```

The folder should contain one subfolder per shape (e.g. `Circle`, `Square`) with images named like:

```
img25-C-4.png
img2298-C-3(110100).png
```

---

## Training

```bash
# Train all shapes with both models
python train.py

# Train a single shape
python train.py --shape Circle

# Train with a specific model
python train.py --shape Circle --model vit_s
python train.py --shape Circle --model mobile
```

Results are saved incrementally to `results/training_summary.json`. Re-running skips already completed shape/model combinations.

---

## Evaluation

```bash
python evaluate.py
# or specify a custom results file
python evaluate.py --json results/training_summary.json
```

Outputs per-shape metrics (Accuracy, MAE, Adjacent Accuracy, QWK), confusion matrices, training curves, and a LaTeX table — all saved to `results/`.

---

## Figures

```bash
# Publication figures → report_overleaf/figures/
python make_figures_v2.py --json results/training_summary.json

# Result figures for the report
python make_results_figs.py
```

---

## Models

Both models use a frozen backbone with a trained classification head:

- **ViT-S/16** — head LR `1e-4`, CutMix + MixUp + DiffuseMix augmentation, mild class-weighted loss
- **MobileViT-S** — head LR `3e-3`, DiffuseMix only, unweighted loss

---

## Loss

Supports standard cross-entropy and **CORN ordinal loss** (recommended), which encodes the rank ordering of scores and improves MAE and QWK.

Set `USE_ORDINAL_LOSS = True` in `config.py` to enable it.