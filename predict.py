"""Detect faces and predict age and gender on an image or webcam stream.

    python predict.py --model checkpoints/final.keras --image group.jpg
    python predict.py --model checkpoints/final.keras --webcam
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np
import tensorflow as tf

from agegender.utkface import GENDERS

detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def crop_with_margin(frame, box, margin: float = 0.25):
    x, y, w, h = box
    mx, my = int(w * margin), int(h * margin)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(frame.shape[1], x + w + mx), min(frame.shape[0], y + h + my)
    return frame[y0:y1, x0:x1]


def annotate(frame, model, image_size: int):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boxes = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    if len(boxes) == 0:
        return frame
    faces = []
    for box in boxes:
        face = cv2.cvtColor(crop_with_margin(frame, box), cv2.COLOR_BGR2RGB)
        faces.append(cv2.resize(face, (image_size, image_size)).astype(np.float32))
    preds = model.predict(np.stack(faces), verbose=0)
    for (x, y, w, h), age, g in zip(boxes, preds["age"].ravel(), preds["gender"].ravel()):
        label = f"{GENDERS[int(g >= 0.5)]}, {age:.0f}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
    return frame


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--image")
    p.add_argument("--webcam", action="store_true")
    p.add_argument("--out", default="prediction.jpg")
    p.add_argument("--image-size", type=int, default=160)
    args = p.parse_args()

    model = tf.keras.models.load_model(args.model)
    if args.image:
        frame = cv2.imread(args.image)
        cv2.imwrite(args.out, annotate(frame, model, args.image_size))
        print(f"Saved {args.out}")
    elif args.webcam:
        cap = cv2.VideoCapture(0)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imshow("Age & gender (q to quit)", annotate(frame, model, args.image_size))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        p.error("pass --image or --webcam")


if __name__ == "__main__":
    main()
