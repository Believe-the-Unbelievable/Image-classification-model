# model.py
# ─────────────────────────────────────────
# Builds the Transfer Learning model.
# Uses EfficientNetB0 pre-trained on ImageNet.
# Imported by train.py.
# ─────────────────────────────────────────

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
from config import (
    IMG_SHAPE, NUM_CLASSES,
    DROPOUT_RATE, DENSE_UNITS
)


def build_model():
    """
    Builds and returns the model with:
    - EfficientNetB0 as frozen base
    - Custom classification head on top
    """

    # ── Base model (frozen) ────────────────
    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,          # remove original 1000-class head
        input_shape=IMG_SHAPE
    )
    base_model.trainable = False    # freeze all base layers

    # ── Custom classification head ─────────
    inputs  = tf.keras.Input(shape=IMG_SHAPE)
    x       = base_model(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dropout(DROPOUT_RATE)(x)
    x       = layers.Dense(DENSE_UNITS, activation="relu")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    print("\n── Model Summary ────────────────────────")
    print(f"  Base      : EfficientNetB0 (frozen)")
    print(f"  Head      : GlobalAvgPool → Dropout → Dense({DENSE_UNITS}) → Dense({NUM_CLASSES})")
    print(f"  Trainable params: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
    print("─────────────────────────────────────────\n")

    return model, base_model


def unfreeze_top_layers(model, base_model, num_layers=20):
    """
    Unfreezes the top N layers of the base model for fine-tuning.
    Call this before Phase 2 training.
    """
    base_model.trainable = True
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False

    trainable = sum([1 for l in base_model.layers if l.trainable])
    print(f"\n── Fine-tuning: {trainable} layers unfrozen in base model ──\n")

    return model