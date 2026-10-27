"""UTKFace dataset helpers (pure Python, no TensorFlow import).

UTKFace file names encode labels as ``{age}_{gender}_{race}_{timestamp}.jpg``
with gender 0 = male, 1 = female. A handful of files in the public release
are missing fields, so parsing is defensive and such files are skipped.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

GENDERS = ("male", "female")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class Sample:
    path: str
    age: int
    gender: int


def parse_filename(name: str) -> tuple[int, int] | None:
    """Return ``(age, gender)`` from a UTKFace file name, or None if malformed."""
    stem = Path(name).name.split(".")[0]
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    try:
        age, gender = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    if not 0 <= age <= 116 or gender not in (0, 1):
        return None
    return age, gender


def list_samples(root: str | Path, min_age: int = 1, max_age: int = 100) -> list[Sample]:
    samples = []
    for path in sorted(Path(root).rglob("*")):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        parsed = parse_filename(path.name)
        if parsed is None:
            continue
        age, gender = parsed
        if min_age <= age <= max_age:
            samples.append(Sample(str(path), age, gender))
    return samples


def split(samples: list[Sample], val: float = 0.1, test: float = 0.1, seed: int = 42):
    """Shuffle deterministically and split into train / val / test lists."""
    items = list(samples)
    random.Random(seed).shuffle(items)
    n = len(items)
    n_test, n_val = int(n * test), int(n * val)
    return items[n_test + n_val :], items[n_test : n_test + n_val], items[:n_test]


AGE_BUCKETS = ((0, 12), (13, 19), (20, 29), (30, 39), (40, 49), (50, 59), (60, 200))


def age_bucket(age: float) -> str:
    for lo, hi in AGE_BUCKETS:
        if lo <= age <= hi:
            return f"{lo}-{hi}" if hi < 200 else f"{lo}+"
    return "unknown"
