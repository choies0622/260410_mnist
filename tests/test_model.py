import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.model import MLP


class TestMLP(unittest.TestCase):
    def test_forward_returns_probability_rows(self):
        model = MLP(input_size=4, hidden_size=3, output_size=2, seed=7)
        x = np.array(
            [
                [0.0, 0.5, 1.0, 0.25],
                [1.0, 0.0, 0.25, 0.75],
            ],
            dtype=np.float32,
        )

        probabilities = model.forward(x)

        self.assertEqual(probabilities.shape, (2, 2))
        np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2), rtol=1e-6)
        self.assertTrue(np.all(probabilities >= 0.0))
        self.assertTrue(np.all(probabilities <= 1.0))

    def test_save_and_load_preserve_predictions(self):
        model = MLP(input_size=4, hidden_size=3, output_size=2, seed=11)
        x = np.array(
            [
                [0.2, 0.3, 0.4, 0.5],
                [0.9, 0.1, 0.0, 0.8],
            ],
            dtype=np.float32,
        )
        expected = model.predict(x)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.npz"
            model.save(path)
            loaded = MLP.load(path)

        np.testing.assert_array_equal(loaded.predict(x), expected)


if __name__ == "__main__":
    unittest.main()
