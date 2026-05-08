"""Evaluate a saved MNIST MLP."""

from __future__ import annotations

import argparse

import numpy as np

from .mnist_loader import load_mnist
from .model import MLP
from .train import accuracy


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: int = 10) -> np.ndarray:
    matrix = np.zeros((classes, classes), dtype=np.int64)
    for actual, predicted in zip(y_true, y_pred, strict=True):
        matrix[int(actual), int(predicted)] += 1
    return matrix


def evaluate(data_dir: str = "_data", model_path: str = "models/mnist_mlp.npz") -> None:
    _, _, x_test, y_test = load_mnist(data_dir)
    model = MLP.load(model_path)
    predictions = model.predict(x_test)
    test_accuracy = accuracy(model, x_test, y_test)
    matrix = confusion_matrix(y_test, predictions)

    print(f"test_accuracy={test_accuracy:.4f}")
    print("confusion_matrix rows=true labels, columns=predicted labels")
    print(matrix)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved MNIST MLP.")
    parser.add_argument("--data-dir", default="_data")
    parser.add_argument("--model-path", default="models/mnist_mlp.npz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(data_dir=args.data_dir, model_path=args.model_path)


if __name__ == "__main__":
    main()
