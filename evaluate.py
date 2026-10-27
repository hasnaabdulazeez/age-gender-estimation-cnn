"""Evaluate a trained model on the held-out test split.

Reports overall age MAE, gender accuracy and age MAE per age bucket, since
errors are usually much larger for children and older adults.

    python evaluate.py --model checkpoints/final.keras --split checkpoints/test_split.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import tensorflow as tf

from agegender.data import make_dataset
from agegender.utkface import Sample, age_bucket


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--image-size", type=int, default=160)
    args = p.parse_args()

    samples = [Sample(**s) for s in json.loads(open(args.split).read())]
    model = tf.keras.models.load_model(args.model)
    preds = model.predict(make_dataset(samples, args.image_size), verbose=1)

    ages = np.array([s.age for s in samples], dtype=float)
    genders = np.array([s.gender for s in samples])
    pred_age = preds["age"].ravel()
    pred_gender = (preds["gender"].ravel() >= 0.5).astype(int)

    print(f"Age MAE:          {np.mean(np.abs(pred_age - ages)):.2f} years")
    print(f"Gender accuracy:  {np.mean(pred_gender == genders):.3f}")

    errors = defaultdict(list)
    for true, pred in zip(ages, pred_age):
        errors[age_bucket(true)].append(abs(pred - true))
    print("\nAge MAE by bucket:")
    for bucket, errs in errors.items():
        print(f"  {bucket:>6}: {np.mean(errs):5.2f}  (n={len(errs)})")


if __name__ == "__main__":
    main()
