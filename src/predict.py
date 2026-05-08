"""Predict MNIST digits from a test-set index or an image file."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from .mnist_loader import load_mnist
from .model import MLP


def prepare_image(path: str | Path) -> np.ndarray:
    image = Image.open(path).convert("L").resize((28, 28))
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    if float(np.mean(pixels)) > 0.5:
        pixels = 1.0 - pixels
    return pixels.reshape(1, 784)


def predict_index(index: int, data_dir: str, model: MLP) -> None:
    _, _, x_test, y_test = load_mnist(data_dir)
    if index < 0 or index >= x_test.shape[0]:
        raise ValueError(f"--index must be between 0 and {x_test.shape[0] - 1}")

    probabilities = model.forward(x_test[index : index + 1])[0]
    predicted = int(np.argmax(probabilities))
    actual = int(y_test[index])
    confidence = float(probabilities[predicted])
    print(f"index={index} predicted={predicted} actual={actual} confidence={confidence:.4f}")


def predict_image(image_path: str, model: MLP) -> None:
    x = prepare_image(image_path)
    probabilities = model.forward(x)[0]
    predicted = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted])
    print(f"image={image_path} predicted={predicted} confidence={confidence:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a digit with a saved MNIST MLP.")
    parser.add_argument("--data-dir", default="_data")
    parser.add_argument("--model-path", default="models/mnist_mlp.npz")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--index", type=int)
    group.add_argument("--image")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = MLP.load(args.model_path)
    if args.index is not None:
        predict_index(args.index, args.data_dir, model)
    else:
        predict_image(args.image, model)


if __name__ == "__main__":
    main()
