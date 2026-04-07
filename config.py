# config.py
# ─────────────────────────────────────────
# All project settings live here.
# Change anything here and it affects every other file automatically.
# ─────────────────────────────────────────

import os

# ── Paths ──────────────────────────────────
RAW_DATA_DIR   = "catdog"        # your collected photos go here
DATASET_DIR    = "dataset"           # split output goes here
SAVED_MODEL    = "saved_model/cat_dog_classifier.keras"

TRAIN_DIR      = os.path.join(DATASET_DIR, "train")
VAL_DIR        = os.path.join(DATASET_DIR, "val")
TEST_DIR       = os.path.join(DATASET_DIR, "test")

# ── Dataset split ratios ───────────────────
TRAIN_RATIO    = 0.70
VAL_RATIO      = 0.15
# test gets the remaining 15% automatically
RANDOM_SEED    = 42

# ── Image settings ─────────────────────────
IMG_SIZE       = (224, 224)
IMG_SHAPE      = (224, 224, 3)
BATCH_SIZE     = 32
NUM_CLASSES    = 2
CLASS_NAMES    = ["Cat", "Dog"]    # must match your folder names

# ── Training settings ──────────────────────
PHASE1_EPOCHS  = 20     # train the head only
PHASE2_EPOCHS  = 20     # fine-tuning
PHASE1_LR      = 1e-3   # learning rate for phase 1
PHASE2_LR      = 1e-5   # much smaller for fine-tuning
DROPOUT_RATE   = 0.3
DENSE_UNITS    = 128

# ── Early stopping ─────────────────────────
ES_PATIENCE    = 5      # stop if no improvement for 5 epochs
LR_PATIENCE    = 3      # reduce LR if no improvement for 3 epochs