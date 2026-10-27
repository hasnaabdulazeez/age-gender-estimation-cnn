# Age & Gender Estimation with a Multi-Output CNN

A single Keras model that predicts **age (regression)** and **gender (binary classification)** from a face crop, with a real-time webcam demo. It is trained on the public [UTKFace](https://susanqq.github.io/UTKFace/) dataset (about 23k faces, ages 0–116).

## Approach

```
face crop (160×160) → MobileNetV2 backbone (ImageNet) → GAP → dropout ─┬─ Dense → age (years)
                                                                        └─ Dense → sigmoid → gender
```

- **Two-stage transfer learning.** Stage 1 trains only the heads on a frozen backbone. Stage 2 unfreezes the top 40 layers at a learning rate of 1e-5, with BatchNorm statistics kept frozen.
- **Huber loss for age.** UTKFace age labels are noisy, and Huber is less sensitive to outliers than MSE. The age loss is down-weighted so both heads train at a comparable scale.
- **Augmentation.** Flips, small rotations, zoom and contrast jitter, applied in `tf.data`.
- **Bucketed evaluation.** `evaluate.py` reports MAE per age bucket, because a single overall MAE hides much larger errors on children and older adults.
- **`--backbone custom`** trains a from-scratch 8-layer CNN for comparison.

## Quick start

```bash
pip install -r requirements.txt
# Download UTKFace (aligned & cropped) and extract to data/raw/UTKFace
python train.py --data data/raw/UTKFace --epochs 15 --finetune-epochs 10
python evaluate.py --model checkpoints/final.keras --split checkpoints/test_split.json
python predict.py --model checkpoints/final.keras --image group_photo.jpg
python predict.py --model checkpoints/final.keras --webcam
```

The test split is saved at training time, so evaluation always uses images the model never saw.

## Project structure

```
agegender/
  utkface.py   # filename label parsing, deterministic splits, age buckets (pure Python)
  data.py      # tf.data pipeline with augmentation
  model.py     # shared backbone + age/gender heads, losses and metrics
train.py       # two-stage training with checkpoints, early stopping, LR schedule
evaluate.py    # MAE, accuracy, per-bucket MAE
predict.py     # Haar-cascade face detection + inference on images/webcam
```

## Responsible use

Apparent age and binary gender from appearance are estimates, not facts about a person. UTKFace has known demographic imbalances, so check errors across groups before using this anywhere decisions affect people.

## License

MIT
