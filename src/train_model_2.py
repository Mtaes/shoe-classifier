from pathlib import Path

import torch
from pytorch_lightning import seed_everything
from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall

from datasets import get_shoe_dataset
from models import ShoeClassifier
from utils import get_args, get_dataloader, get_trainer


def freeze_params(model: torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = False
    for _, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        freeze_params(child)


if __name__ == "__main__":
    seed_everything(seed=42, workers=True)
    args = get_args(ckpt=True)
    model = ShoeClassifier().load_from_checkpoint(args.ckpt, lr=args.lr)
    model.model._modules["9"] = torch.nn.Linear(32, 1)
    for i in range(0, 4):
        freeze_params(model.model._modules[str(i)])
    model.metrics = MetricCollection(
        [
            Accuracy(num_classes=2, multiclass=True),
            Precision(num_classes=2, multiclass=True),
            Recall(num_classes=2, multiclass=True),
            F1Score(num_classes=2, multiclass=True),
        ]
    )

    data_path = Path("..") / "data" / "Shoes"
    train_dataset = get_shoe_dataset(str(data_path / "train"), train=True)
    val_dataset = get_shoe_dataset(str(data_path / "test"))
    train_loader = get_dataloader(train_dataset, args, shuffle=True)
    val_loader = get_dataloader(val_dataset, args)

    log_path = Path("..") / "logs"
    trainer = get_trainer(args.epochs, log_path, name="classifier_2")
    trainer.fit(model, train_loader, val_loader)
    trainer.test(dataloaders=val_loader, ckpt_path="best")
