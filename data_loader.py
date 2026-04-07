# data_loader.py  (FIXED)
# ─────────────────────────────────────────
# KEY FIX: EfficientNetB0 has its OWN built-in preprocessing.
# Do NOT use rescale=1./255 — it corrupts the pixel values
# and causes the model to predict one class for everything.
# ─────────────────────────────────────────

import os
import tensorflow as tf
from config import (
    TRAIN_DIR, VAL_DIR, TEST_DIR,
    IMG_SIZE, BATCH_SIZE
)


def get_data_generators():
    """
    Returns three generators: train_gen, val_gen, test_gen.
    - train_gen  : augmented (flip, rotate, zoom, shift) — NO rescale
    - val_gen    : no augmentation, no rescale
    - test_gen   : no augmentation, shuffle=False (important for evaluation)
    """

    # ── Training augmentation — NO rescale ─
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        horizontal_flip=True,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
        # ✅ NO rescale=1./255 — EfficientNetB0 preprocesses internally
    )

    # ── Val / Test — no augmentation, no rescale ──
    val_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True
    )

    val_gen = val_test_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    test_gen = val_test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False      # must be False for evaluation to work correctly
    )

    print("\n── Data Loaded ──────────────────────────")
    print(f"  Classes   : {train_gen.class_indices}")
    print(f"  Train     : {train_gen.samples} images")
    print(f"  Val       : {val_gen.samples} images")
    print(f"  Test      : {test_gen.samples} images")

    # ── Class balance check ────────────────
    print("\n── Class Balance Check ──────────────────")
    for split, path in [("train", TRAIN_DIR), ("val", VAL_DIR), ("test", TEST_DIR)]:
        counts = {
            cls: len(os.listdir(os.path.join(path, cls)))
            for cls in os.listdir(path)
            if os.path.isdir(os.path.join(path, cls))
        }
        vals = list(counts.values())
        status = "✅" if max(vals) / min(vals) < 1.5 else "⚠️  IMBALANCED — fix this!"
        print(f"  {split:6}: {counts}  {status}")
    print("─────────────────────────────────────────\n")

    return train_gen, val_gen, test_gen