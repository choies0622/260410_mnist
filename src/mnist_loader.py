"""Utilities for reading MNIST IDX gzip files."""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np


TRAIN_IMAGES = "train-images-idx3-ubyte.gz"
TRAIN_LABELS = "train-labels-idx1-ubyte.gz"
TEST_IMAGES = "t10k-images-idx3-ubyte.gz"
TEST_LABELS = "t10k-labels-idx1-ubyte.gz"


def _read_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as file:
        magic, count, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError(f"{path} is not an MNIST image file")
        data = np.frombuffer(file.read(), dtype=np.uint8)

    expected = count * rows * cols
    if data.size != expected:
        raise ValueError(f"{path} has {data.size} image bytes, expected {expected}")

    return data.reshape(count, rows * cols).astype(np.float32) / 255.0


def _read_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as file:
        magic, count = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError(f"{path} is not an MNIST label file")
        labels = np.frombuffer(file.read(), dtype=np.uint8)

    if labels.size != count:
        raise ValueError(f"{path} has {labels.size} labels, expected {count}")

    return labels.astype(np.int64)


def load_mnist(data_dir: str | Path = "_data") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST arrays from the standard four gzip files."""
    data_path = Path(data_dir)
    x_train = _read_images(data_path / TRAIN_IMAGES)
    y_train = _read_labels(data_path / TRAIN_LABELS)
    x_test = _read_images(data_path / TEST_IMAGES)
    y_test = _read_labels(data_path / TEST_LABELS)
    return x_train, y_train, x_test, y_test
