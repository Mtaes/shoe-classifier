from pathlib import Path

from pytorch_lightning import seed_everything

from datasets import FashionMNISTDataset
from models import ShoeClassifier
from utils import get_args, get_dataloader, get_trainer

if __name__ == "__main__":
    seed_everything(seed=42, workers=True)
    args = get_args()
    data_path = Path("..") / "data" / "fashion-mnist-prep"
    train_dataset = FashionMNISTDataset(data_path / "train", train=True)
    val_dataset = FashionMNISTDataset(data_path / "val")
    test_dataset = FashionMNISTDataset(data_path / "test")

    train_loader = get_dataloader(train_dataset, args, shuffle=True)
    val_loader = get_dataloader(val_dataset, args)
    test_loader = get_dataloader(test_dataset, args)

    log_path = Path("..") / "logs"
    model = ShoeClassifier(lr=args.lr)
    trainer = get_trainer(args.epochs, log_path, name="classifier_1")
    trainer.fit(model, train_loader, val_loader)
    trainer.test(dataloaders=test_loader, ckpt_path="best")
