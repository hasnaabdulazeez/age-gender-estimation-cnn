"""Multi-output CNN: one shared backbone, an age regression head and a gender head."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers


def _custom_backbone(inputs):
    x = layers.Rescaling(1.0 / 255)(inputs)
    for filters in (32, 64, 128, 256):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
    return layers.GlobalAveragePooling2D()(x), None


def _mobilenet_backbone(inputs):
    base = tf.keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", input_shape=tuple(inputs.shape[1:])
    )
    base.trainable = False
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    return layers.GlobalAveragePooling2D()(x), base


def build_model(image_size: int = 160, backbone: str = "mobilenetv2", dropout: float = 0.3):
    """Return ``(model, base)``; ``base`` is the frozen pretrained backbone (or None)."""
    inputs = layers.Input((image_size, image_size, 3))
    if backbone == "mobilenetv2":
        features, base = _mobilenet_backbone(inputs)
    elif backbone == "custom":
        features, base = _custom_backbone(inputs)
    else:
        raise ValueError(f"unknown backbone: {backbone}")

    shared = layers.Dropout(dropout)(features)

    age = layers.Dense(128, activation="relu")(shared)
    age = layers.Dense(1, name="age")(age)

    gender = layers.Dense(64, activation="relu")(shared)
    gender = layers.Dense(1, activation="sigmoid", name="gender")(gender)

    model = tf.keras.Model(inputs, {"age": age, "gender": gender}, name=f"age_gender_{backbone}")
    return model, base


def compile_model(model: tf.keras.Model, lr: float = 1e-3, age_weight: float = 0.05) -> None:
    """Huber loss for age (robust to noisy labels), BCE for gender.

    Age loss is in years, so it is down-weighted to keep both heads on a similar scale.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss={"age": tf.keras.losses.Huber(delta=5.0), "gender": "binary_crossentropy"},
        loss_weights={"age": age_weight, "gender": 1.0},
        metrics={"age": ["mae"], "gender": ["accuracy"]},
    )
