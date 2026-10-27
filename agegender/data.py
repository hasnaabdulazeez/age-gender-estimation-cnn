"""tf.data input pipeline with augmentation."""

from __future__ import annotations

import tensorflow as tf

from .utkface import Sample

AUTOTUNE = tf.data.AUTOTUNE


def _load(path, age, gender, image_size: int):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img = tf.image.resize(img, (image_size, image_size))
    return img, {"age": tf.cast(age, tf.float32), "gender": tf.cast(gender, tf.float32)}


_augment = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.15),
    ],
    name="augment",
)


def make_dataset(
    samples: list[Sample],
    image_size: int = 160,
    batch_size: int = 64,
    training: bool = False,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(
        (
            [s.path for s in samples],
            [s.age for s in samples],
            [s.gender for s in samples],
        )
    )
    if training:
        ds = ds.shuffle(len(samples), reshuffle_each_iteration=True)
    ds = ds.map(lambda p, a, g: _load(p, a, g, image_size), num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(lambda x, y: (_augment(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.batch(batch_size).prefetch(AUTOTUNE)
