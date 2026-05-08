import unittest

import numpy as np

from src.mnist_loader import load_mnist


class TestMnistLoader(unittest.TestCase):
    def test_loads_standard_mnist_shapes_and_ranges(self):
        x_train, y_train, x_test, y_test = load_mnist("_data")

        self.assertEqual(x_train.shape, (60000, 784))
        self.assertEqual(y_train.shape, (60000,))
        self.assertEqual(x_test.shape, (10000, 784))
        self.assertEqual(y_test.shape, (10000,))

        self.assertEqual(x_train.dtype, np.float32)
        self.assertEqual(x_test.dtype, np.float32)
        self.assertTrue(np.all(x_train >= 0.0))
        self.assertTrue(np.all(x_train <= 1.0))
        self.assertTrue(np.all(x_test >= 0.0))
        self.assertTrue(np.all(x_test <= 1.0))

        self.assertTrue(np.issubdtype(y_train.dtype, np.integer))
        self.assertTrue(np.issubdtype(y_test.dtype, np.integer))


if __name__ == "__main__":
    unittest.main()
