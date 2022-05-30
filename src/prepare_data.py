import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split


def read_gzip(labels_path: Path, images_path: Path):
    with gzip.open(labels_path, "rb") as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
    with gzip.open(images_path, "rb") as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(
            len(labels), 28 * 28
        )
    return labels, images


def download_files(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    files = (
        (
            "train-images-idx3-ubyte.gz",
            "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
        ),
        (
            "train-labels-idx1-ubyte.gz",
            "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
        ),
        (
            "t10k-images-idx3-ubyte.gz",
            "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
        ),
        (
            "t10k-labels-idx1-ubyte.gz",
            "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
        ),
    )
    for f, url in files:
        if not (root / f).is_file():
            with open(root / f, "wb") as out_file:
                content = requests.get(url, stream=True).content
                out_file.write(content)


if __name__ == "__main__":
    orig_data_path = Path("..") / "data" / "fashion-mnist"
    download_files(orig_data_path)
    out_data_path = Path("..") / "data" / "fashion-mnist-prep"
    labels, images = read_gzip(
        orig_data_path / "train-labels-idx1-ubyte.gz",
        orig_data_path / "train-images-idx3-ubyte.gz",
    )
    test_labels, test_images = read_gzip(
        orig_data_path / "t10k-labels-idx1-ubyte.gz",
        orig_data_path / "t10k-images-idx3-ubyte.gz",
    )

    transformed_labels = np.where(pd.Series(labels).isin([5, 7, 9]), 1, 0).astype(
        np.uint8
    )
    transformed_test_labels = np.where(
        pd.Series(test_labels).isin([5, 7, 9]), 1, 0
    ).astype(np.uint8)

    images = images.reshape(-1, 28, 28)
    test_images = test_images.reshape(-1, 28, 28)

    (
        train_images,
        val_images,
        train_labels,
        val_labels,
        train_tlabels,
        val_tlabels,
    ) = train_test_split(
        images,
        labels,
        transformed_labels,
        test_size=0.3,
        random_state=42,
        stratify=np.concatenate(
            (
                np.expand_dims(labels, axis=1),
                np.expand_dims(transformed_labels, axis=1),
            ),
            axis=1,
        ),
    )

    data_list = [
        [train_images, train_labels, train_tlabels],
        [val_images, val_labels, val_tlabels],
        [test_images, test_labels, transformed_test_labels],
    ]

    for subset, data in zip(["train", "val", "test"], data_list):
        prep_data_path = out_data_path / subset
        prep_data_path.mkdir(parents=True, exist_ok=True)
        with open(prep_data_path / "images.npy", mode="wb") as f:
            np.save(f, data[0])
        with open(prep_data_path / "labels.npy", mode="wb") as f:
            np.save(f, data[1])
        with open(prep_data_path / "tlabels.npy", mode="wb") as f:
            np.save(f, data[2])
