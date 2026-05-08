"""Train the MNIST MLP."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .mnist_loader import load_mnist
from .model import MLP


def accuracy(model: MLP, x: np.ndarray, y: np.ndarray, batch_size: int = 1024) -> float:
    correct = 0
    for start in range(0, x.shape[0], batch_size):
        end = start + batch_size
        correct += int(np.sum(model.predict(x[start:end]) == y[start:end]))
    return correct / y.size


def train(
    data_dir: str = "_data",
    model_path: str = "models/mnist_mlp.npz",
    epochs: int = 5,
    batch_size: int = 128,
    learning_rate: float = 0.1,
    hidden_size: int = 128,
    seed: int = 42,
) -> MLP:
    x_train, y_train, x_test, y_test = load_mnist(data_dir)
    model = MLP(hidden_size=hidden_size, seed=seed)
    rng = np.random.default_rng(seed)

    for epoch in range(1, epochs + 1):
        indices = rng.permutation(x_train.shape[0])
        epoch_loss = 0.0
        batches = 0

        for start in range(0, x_train.shape[0], batch_size):
            batch_indices = indices[start : start + batch_size]
            x_batch = x_train[batch_indices]
            y_batch = y_train[batch_indices]

            probabilities = model.forward(x_batch)
            batch_loss = model.loss(probabilities, y_batch)
            model.backward(probabilities, y_batch)
            model.step(learning_rate)

            epoch_loss += batch_loss
            batches += 1

        test_accuracy = accuracy(model, x_test, y_test)
        print(
            f"epoch {epoch}/{epochs} "
            f"loss={epoch_loss / batches:.4f} "
            f"test_accuracy={test_accuracy:.4f}"
        )

    model.save(model_path)
    print(f"saved model to {Path(model_path)}")
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a NumPy MLP on MNIST.")
    parser.add_argument("--data-dir", default="_data")
    parser.add_argument("--model-path", default="models/mnist_mlp.npz")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        data_dir=args.data_dir,
        model_path=args.model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
