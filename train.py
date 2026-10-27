"""Train the age/gender model on UTKFace in two stages.

Stage 1 trains only the heads on a frozen ImageNet backbone.
Stage 2 unfreezes the top of the backbone and fine-tunes with a lower learning rate.

    python train.py --data data/raw/UTKFace --epochs 15 --finetune-epochs 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf

from agegender.data import make_dataset
from agegender.model import build_model, compile_model
from agegender.utkface import list_samples, split


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--backbone", default="mobilenetv2", choices=["mobilenetv2", "custom"])
    p.add_argument("--image-size", type=int, default=160)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--finetune-epochs", type=int, default=10)
    p.add_argument("--unfreeze-last", type=int, default=40, help="backbone layers to unfreeze in stage 2")
    p.add_argument("--out", default="checkpoints")
    args = p.parse_args()

    samples = list_samples(args.data)
    if not samples:
        raise SystemExit(f"No UTKFace images found under {args.data}")
    train_s, val_s, test_s = split(samples)
    print(f"train={len(train_s)} val={len(val_s)} test={len(test_s)}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "test_split.json").write_text(json.dumps([s.__dict__ for s in test_s]))

    train_ds = make_dataset(train_s, args.image_size, args.batch_size, training=True)
    val_ds = make_dataset(val_s, args.image_size, args.batch_size)

    model, base = build_model(args.image_size, args.backbone)
    compile_model(model, lr=1e-3)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(out / "best.keras"), monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
        tf.keras.callbacks.CSVLogger(str(out / "history.csv"), append=True),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    if base is not None and args.finetune_epochs > 0:
        base.trainable = True
        for layer in base.layers[: -args.unfreeze_last]:
            layer.trainable = False
        for layer in base.layers:  # keep BatchNorm statistics frozen while fine-tuning
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        compile_model(model, lr=1e-5)
        model.fit(train_ds, validation_data=val_ds, epochs=args.finetune_epochs, callbacks=callbacks)

    model.save(out / "final.keras")
    print(f"Saved model to {out / 'final.keras'}")


if __name__ == "__main__":
    main()
