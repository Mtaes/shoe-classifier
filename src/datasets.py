from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class FashionMNISTDataset(Dataset):
    def __init__(self, data_path: Path):
        with open(data_path / "images.npy", mode="rb") as f:
            self.images = np.load(f)
        with open(data_path / "tlabels.npy", mode="rb") as f:
            self.labels = np.load(f)

    def __getitem__(self, idx):
        image = (
            torch.tensor(self.images[idx].reshape(1, 28, 28), dtype=torch.float32) / 255
        )
        target = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, target

    def __len__(self):
        return len(self.labels)