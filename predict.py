# predict.py
# ─────────────────────────────────────────
# Predicts the class of any new image using the trained model.
# ─────────────────────────────────────────
# Usage:
#   python predict.py
#   python predict.py --image path/to/your/photo.jpg
# ─────────────────────────────────────────

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from config import SAVED_MODEL, CLASS_NAMES, IMG_SIZE


def predict_image(img_path, model):
    """Load an image, run prediction, and display the result."""

    # ── Load & preprocess ──────────────────
    img = tf.keras.preprocessing.image.load_img(
        img_path,
        target_size=IMG_SIZE
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    # ✅ Do NOT divide by 255 — EfficientNetB0 has built-in preprocessing.
    # Training generators also had no rescale=1./255, so we must match that here.
    img_array = np.expand_dims(img_array, axis=0)   # add batch dimension  (raw 0–255)

    # ── Predict ────────────────────────────
    predictions  = model.predict(img_array, verbose=0)
    predicted_idx = int(np.argmax(predictions[0]))
    confidence    = float(np.max(predictions[0])) * 100

    predicted_class = CLASS_NAMES[predicted_idx]

    # ── Print result ───────────────────────
    print("\n── Prediction Result ────────────────────")
    print(f"  Image      : {img_path}")
    print(f"  Prediction : {predicted_class.upper()}")
    print(f"  Confidence : {confidence:.1f}%")
    print("\n  All class probabilities:")
    for i, class_name in enumerate(CLASS_NAMES):
        bar   = "█" * int(predictions[0][i] * 30)
        print(f"  {class_name:10} : {bar:30} {predictions[0][i]*100:.1f}%")
    print("─────────────────────────────────────────")

    # ── Show image ─────────────────────────
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title(
        f"Predicted: {predicted_class.upper()}  ({confidence:.1f}%)",
        fontsize=13,
        color="green" if confidence > 80 else "orange"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return predicted_class, confidence


def main():
    parser = argparse.ArgumentParser(description="Predict cat or dog from an image.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to the image file"
    )
    args = parser.parse_args()

    # ── Load model ─────────────────────────
    import os
    if not os.path.exists(SAVED_MODEL):
        print(f"ERROR: No saved model found at '{SAVED_MODEL}'")
        print("Run  python train.py  first.")
        sys.exit(1)

    print(f"Loading model from: {SAVED_MODEL}")
    model = tf.keras.models.load_model(SAVED_MODEL)
    print("✓ Model loaded!\n")

    # ── Get image path ─────────────────────
    if args.image:
        img_path = args.image
    else:
        img_path = input("Enter path to your image: ").strip()

    if not os.path.exists(img_path):
        print(f"ERROR: Image not found at '{img_path}'")
        sys.exit(1)

    predict_image(img_path, model)


if __name__ == "__main__":
    main()