from pathlib import Path

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from datasets import FashionMNISTDataset
from models import ShoeClassifier

if __name__ == "__main__":
    seed_everything(seed=42, workers=True)
    data_path = Path("..") / "data" / "fashion-mnist-prep"
    train_dataset = FashionMNISTDataset(data_path / "train")
    val_dataset = FashionMNISTDataset(data_path / "val")
    test_dataset = FashionMNISTDataset(data_path / "test")

    batch = 4096
    workers = 4
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch,
        num_workers=workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch,
        num_workers=workers,
        pin_memory=True,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch,
        num_workers=workers,
        pin_memory=True,
        shuffle=False,
    )

    log_path = Path("..") / "logs"
    model = ShoeClassifier()
    trainer = Trainer(
        default_root_dir=str(log_path),
        gpus=1 if torch.cuda.is_available() else None,
        max_epochs=200,
        logger=TensorBoardLogger(log_path, name="classifier_1"),
        log_every_n_steps=1,
        callbacks=[ModelCheckpoint(monitor="epoch_val/loss")],
        deterministic=True,
        detect_anomaly=True,
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(dataloaders=test_loader, ckpt_path="best")
