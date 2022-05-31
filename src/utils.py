from argparse import ArgumentParser
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset


def get_args(ckpt: bool = False):
    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    if ckpt:
        parser.add_argument("--ckpt", type=Path, required=True)
    args = parser.parse_args()
    return args


def get_dataloader(data: Dataset, args, shuffle: bool = False):
    return DataLoader(
        data,
        batch_size=args.batch,
        num_workers=args.workers,
        pin_memory=True,
        shuffle=shuffle,
        persistent_workers=True,
    )


def get_trainer(epochs: int, log_path: Path, name: str):
    return Trainer(
        default_root_dir=str(log_path),
        gpus=1 if torch.cuda.is_available() else None,
        max_epochs=epochs,
        logger=TensorBoardLogger(log_path, name=name),
        log_every_n_steps=1,
        callbacks=[ModelCheckpoint(monitor="epoch_val/loss")],
        deterministic=True,
        detect_anomaly=True,
    )
