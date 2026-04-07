# evaluate.py
# ─────────────────────────────────────────
# Evaluates the trained model on the TEST set.
# Run this only once after training is complete.
# ─────────────────────────────────────────
# Usage:
#   python evaluate.py
# ─────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from config import SAVED_MODEL, CLASS_NAMES
from data_loader import get_data_generators


def evaluate():
    print("\n" + "=" * 45)
    print("       FINAL TEST SET EVALUATION")
    print("=" * 45)

    # ── Load model ─────────────────────────
    if not __import__("os").path.exists(SAVED_MODEL):
        print(f"ERROR: No saved model found at '{SAVED_MODEL}'")
        print("Run  python train.py  first.")
        return

    model = tf.keras.models.load_model(SAVED_MODEL)
    print(f"✓ Model loaded from: {SAVED_MODEL}")

    # ── Load test data ─────────────────────
    _, _, test_gen = get_data_generators()

    # ── Evaluate ───────────────────────────
    print("Running evaluation on test set...")
    test_loss, test_acc = model.evaluate(test_gen, verbose=1)

    print("\n── Results ──────────────────────────────")
    print(f"  Test Accuracy : {test_acc  * 100:.2f}%")
    print(f"  Test Loss     : {test_loss:.4f}")
    print("─────────────────────────────────────────")

    # ── Predictions ────────────────────────
    y_true = test_gen.classes
    y_pred = np.argmax(model.predict(test_gen, verbose=1), axis=1)

    # ── Classification report ──────────────
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # ── Confusion matrix ───────────────────
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )
    plt.title("Confusion Matrix — Test Set")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()
    print("  Confusion matrix saved: confusion_matrix.png")

    # ── Per-class accuracy ─────────────────
    print("\nPer-class accuracy:")
    for i, class_name in enumerate(CLASS_NAMES):
        class_total   = cm[i].sum()
        class_correct = cm[i][i]
        print(f"  {class_name:10} : {class_correct}/{class_total}  ({class_correct/class_total*100:.1f}%)")

    print("\nNext step → run:  python predict.py")


if __name__ == "__main__":
    evaluate()