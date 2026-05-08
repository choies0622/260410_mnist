"""A small NumPy MLP for MNIST classification."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class MLP:
    def __init__(
        self,
        input_size: int = 784,
        hidden_size: int = 128,
        output_size: int = 10,
        seed: int = 42,
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.seed = seed

        rng = np.random.default_rng(seed)
        self.w1 = (rng.standard_normal((input_size, hidden_size)) * np.sqrt(2.0 / input_size)).astype(
            np.float32
        )
        self.b1 = np.zeros(hidden_size, dtype=np.float32)
        self.w2 = (rng.standard_normal((hidden_size, output_size)) * np.sqrt(2.0 / hidden_size)).astype(
            np.float32
        )
        self.b2 = np.zeros(output_size, dtype=np.float32)
        self._cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._grads: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        hidden_linear = x @ self.w1 + self.b1
        hidden = np.maximum(hidden_linear, 0.0)
        logits = hidden @ self.w2 + self.b2
        probabilities = self._softmax(logits)
        self._cache = (x, hidden_linear, hidden)
        return probabilities

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(x), axis=1)

    def loss(self, probabilities: np.ndarray, labels: np.ndarray) -> float:
        labels = labels.astype(np.int64, copy=False)
        clipped = np.clip(probabilities[np.arange(labels.size), labels], 1e-12, 1.0)
        return float(-np.mean(np.log(clipped)))

    def backward(self, probabilities: np.ndarray, labels: np.ndarray) -> None:
        if self._cache is None:
            raise RuntimeError("forward must be called before backward")

        x, hidden_linear, hidden = self._cache
        labels = labels.astype(np.int64, copy=False)
        batch_size = labels.size

        dlogits = probabilities.copy()
        dlogits[np.arange(batch_size), labels] -= 1.0
        dlogits /= batch_size

        dw2 = hidden.T @ dlogits
        db2 = np.sum(dlogits, axis=0)
        dhidden = dlogits @ self.w2.T
        dhidden[hidden_linear <= 0.0] = 0.0
        dw1 = x.T @ dhidden
        db1 = np.sum(dhidden, axis=0)

        self._grads = (
            dw1.astype(np.float32),
            db1.astype(np.float32),
            dw2.astype(np.float32),
            db2.astype(np.float32),
        )

    def step(self, learning_rate: float) -> None:
        if self._grads is None:
            raise RuntimeError("backward must be called before step")

        dw1, db1, dw2, db2 = self._grads
        self.w1 -= learning_rate * dw1
        self.b1 -= learning_rate * db1
        self.w2 -= learning_rate * dw2
        self.b2 -= learning_rate * db2

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            input_size=np.array(self.input_size),
            hidden_size=np.array(self.hidden_size),
            output_size=np.array(self.output_size),
            seed=np.array(self.seed),
            w1=self.w1,
            b1=self.b1,
            w2=self.w2,
            b2=self.b2,
        )

    @classmethod
    def load(cls, path: str | Path) -> "MLP":
        data = np.load(path)
        model = cls(
            input_size=int(data["input_size"]),
            hidden_size=int(data["hidden_size"]),
            output_size=int(data["output_size"]),
            seed=int(data["seed"]),
        )
        model.w1 = data["w1"].astype(np.float32)
        model.b1 = data["b1"].astype(np.float32)
        model.w2 = data["w2"].astype(np.float32)
        model.b2 = data["b2"].astype(np.float32)
        return model
