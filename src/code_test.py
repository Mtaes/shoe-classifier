import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from datasets import FashionMNISTDataset
from models import ShoeClassifier


class TestFashionMNISTDataset(unittest.TestCase):
    def setUp(self):
        self.data_path = tempfile.mkdtemp()
        np.random.seed(42)
        self.images = np.random.randint(
            low=0, high=256, size=(3, 28, 28), dtype=np.uint8
        )
        self.labels = np.random.randint(low=0, high=2, size=(3), dtype=np.uint8)
        with open(Path(self.data_path) / "images.npy", mode="wb") as f:
            np.save(f, self.images)
        with open(Path(self.data_path) / "tlabels.npy", mode="wb") as f:
            np.save(f, self.labels)

    def tearDown(self):
        shutil.rmtree(self.data_path)

    def test_image_shape(self):
        dataset = FashionMNISTDataset(Path(self.data_path))
        image, _ = dataset[0]
        self.assertEqual(image.shape, (1, 28, 28))

    def test_image_values(self):
        dataset = FashionMNISTDataset(Path(self.data_path))
        image, _ = dataset[0]
        example_image = (self.images[0].flatten().astype(np.float32) / 255).tolist()
        self.assertEqual(example_image, image.flatten().tolist())

    def test_image_dtype(self):
        dataset = FashionMNISTDataset(Path(self.data_path))
        image, _ = dataset[0]
        self.assertTrue(type(image) is torch.Tensor)
        self.assertEqual(image.dtype, torch.float32)

    def test_target_value(self):
        dataset = FashionMNISTDataset(Path(self.data_path))
        _, target = dataset[0]
        self.assertEqual(target, self.labels[0])

    def test_target_dtype(self):
        dataset = FashionMNISTDataset(Path(self.data_path))
        _, target = dataset[0]
        self.assertTrue(type(target) is torch.Tensor)
        self.assertEqual(target.dtype, torch.float32)


class TestShoeClassifier(unittest.TestCase):
    def test_model_output(self):
        torch.manual_seed(42)
        input_data = torch.rand((7, 1, 28, 28))
        model = ShoeClassifier()
        model.eval()
        with torch.no_grad():
            output = model(input_data)
        self.assertEqual(output.shape, (7, 1))


if __name__ == "__main__":
    unittest.main()
