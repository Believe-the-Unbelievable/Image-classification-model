# train.py  (FIXED)
# ─────────────────────────────────────────
# Main training script. Run this to train your model.
# Handles Phase 1 (head only) and Phase 2 (fine-tuning).
# ─────────────────────────────────────────
# Usage:
#   python train.py
# ─────────────────────────────────────────

import os
import matplotlib.pyplot as plt
import tensorflow as tf

from config import (
    PHASE1_EPOCHS, PHASE2_EPOCHS,
    PHASE1_LR, PHASE2_LR,
    ES_PATIENCE, LR_PATIENCE,
    SAVED_MODEL
)
from data_loader import get_data_generators
from model import build_model, unfreeze_top_layers


def get_callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=ES_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=LR_PATIENCE,
            min_lr=1e-7,
            verbose=1
        )
    ]


def plot_history(history, title, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["accuracy"],     label="Train")
    ax1.plot(history.history["val_accuracy"], label="Val")
    ax1.set_title(f"{title} — Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim([0, 1])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history["loss"],     label="Train")
    ax2.plot(history.history["val_loss"], label="Val")
    ax2.set_title(f"{title} — Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    print(f"  Plot saved: {filename}")


def check_data_balance(train_gen):
    """Warn if classes are imbalanced."""
    counts = {}
    for cls, idx in train_gen.class_indices.items():
        counts[cls] = sum(1 for l in train_gen.classes if l == idx)
    vals = list(counts.values())
    ratio = max(vals) / min(vals)
    if ratio > 1.5:
        print(f"\n⚠️  WARNING: Class imbalance detected! {counts}")
        print("   This can cause the model to predict only one class.")
        print("   Fix: make sure each class has a similar number of images.\n")
    else:
        print(f"✅ Classes are balanced: {counts}\n")


def train():
    print("\n" + "=" * 45)
    print("    CAT vs DOG CLASSIFIER — TRAINING")
    print("=" * 45)

    # ── Load data ──────────────────────────
    train_gen, val_gen, _ = get_data_generators()
    check_data_balance(train_gen)

    # ── Build model ────────────────────────
    model, base_model = build_model()

    # ══════════════════════════════════════
    # PHASE 1 — Train head only
    # ══════════════════════════════════════
    print("PHASE 1: Training classification head only...")
    print(f"  Epochs : up to {PHASE1_EPOCHS}  |  LR: {PHASE1_LR}")
    print("-" * 45)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE1_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history1 = model.fit(
        train_gen,
        epochs=PHASE1_EPOCHS,
        validation_data=val_gen,
        callbacks=get_callbacks()
    )

    best_val = max(history1.history["val_accuracy"])
    print(f"\n  Phase 1 best val accuracy: {best_val*100:.2f}%")

    if best_val < 0.55:
        print("\n⚠️  WARNING: Val accuracy is still near 50%.")
        print("   Possible causes:")
        print("   1. Class imbalance in your dataset")
        print("   2. Too few images (need 200+ per class)")
        print("   3. Corrupted or mislabeled images")
        print("   Check the class balance printed above.\n")

    plot_history(history1, "Phase 1 (Head Training)", "phase1_results.png")

    # ══════════════════════════════════════
    # PHASE 2 — Fine-tune top layers
    # ══════════════════════════════════════
    print("\nPHASE 2: Fine-tuning top 20 layers...")
    print(f"  Epochs : up to {PHASE2_EPOCHS}  |  LR: {PHASE2_LR}  (100x smaller!)")
    print("-" * 45)

    model = unfreeze_top_layers(model, base_model, num_layers=20)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE2_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history2 = model.fit(
        train_gen,
        epochs=PHASE2_EPOCHS,
        validation_data=val_gen,
        callbacks=get_callbacks()
    )

    best_val2 = max(history2.history["val_accuracy"])
    print(f"\n  Phase 2 best val accuracy: {best_val2*100:.2f}%")
    plot_history(history2, "Phase 2 (Fine-tuning)", "phase2_results.png")

    # ── Save ───────────────────────────────
    os.makedirs(os.path.dirname(SAVED_MODEL), exist_ok=True)
    model.save(SAVED_MODEL)
    print(f"\n✓ Model saved: {SAVED_MODEL}")
    print("Next → run:  python evaluate.py")


if __name__ == "__main__":
    train()