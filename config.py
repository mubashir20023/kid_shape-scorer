"""
config.py — Central configuration for the BOT-2 shape scoring pipeline.
"""

import os

DATASET_PATH = r"C:\Users\Administrator\Downloads\Shapes\Shapes"

SHAPES = {
    "Circle":            {"code": "C",  "max_score": 4},
    "Square":            {"code": "S",  "max_score": 5},
    "Triangle":          {"code": "T",  "max_score": 5},
    "Diagonal":          {"code": "D",  "max_score": 5},
    "Wave":              {"code": "W",  "max_score": 4},
    "Star":              {"code": "ST", "max_score": 5},
    "Overlapped circle": {"code": "O",  "max_score": 6},
    "Overlapped pencils":{"code": "p",  "max_score": 6},
}

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SKIP_FILES = {"img1688-ST-6.png"}

# Image / DataLoader
IMG_SIZE    = 224
BATCH_SIZE  = 32
NUM_WORKERS = 0

# Training schedule
NUM_EPOCHS       = 25
FREEZE_EPOCHS    = 5
HEAD_LR          = 1e-3
BACKBONE_LR      = 1e-5
LR               = 1e-4
WARMUP_EPOCHS    = 3
WEIGHT_DECAY     = 1e-2
GRAD_ACCUM_STEPS = 2

# Loss
USE_ORDINAL_LOSS = True
LABEL_SMOOTHING  = 0.1

# Augmentation
CUTMIX_PROB     = 0.5
MIXUP_PROB      = 0.3
DIFFUSEMIX_PROB = 0.4
MIXUP_ALPHA     = 0.4
CUTMIX_ALPHA    = 1.0

# Regularisation
USE_TTA   = True
USE_EMA   = True
EMA_DECAY = 0.9998

USE_SWA         = True
SWA_START_EPOCH = 18
SWA_LR          = 5e-5

# Data splitting
VAL_SPLIT       = 0.2
TEST_SPLIT      = 0.1
MIN_CLASS_COUNT = 1

SEED = 42