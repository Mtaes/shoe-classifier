from typing import Any, Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall


class ShoeClassifier(pl.LightningModule):
    def __init__(self, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.model = nn.Sequential(
            nn.Conv2d(1, 24, 3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(24, 48, 3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(48, 64, 3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        metrics = MetricCollection(
            [
                Accuracy(num_classes=1, multiclass=False),
                Precision(num_classes=1, multiclass=False),
                Recall(num_classes=1, multiclass=False),
                F1Score(num_classes=1, multiclass=False),
            ]
        )
        self.loss = nn.BCELoss()
        self.train_metrics = metrics.clone(prefix="train/")
        self.valid_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, metrics: MetricCollection):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y.reshape(-1, 1))
        batch_metrics = metrics(y_hat, y.reshape(-1, 1).to(torch.int))
        batch_metrics = {f"batch_{key}": item for key, item in batch_metrics.items()}
        return loss, batch_metrics

    def training_step(self, batch, batch_idx):
        loss, metrics = self._step(batch, self.train_metrics)
        self.log("batch_train/loss", loss)
        self.log_dict(metrics)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self._step(batch, self.valid_metrics)
        self.log("batch_val/loss", loss)
        self.log_dict(metrics)
        return loss

    def test_step(self, batch, batch_idx):
        loss, metrics = self._step(batch, self.test_metrics)
        self.log("batch_test/loss", loss)
        self.log_dict(metrics)
        return loss

    def _epoch_end(self, step_outputs: list, epoch_metrics: Dict[str, Any]):
        loss = sum(step_outputs) / len(step_outputs)
        epoch_metrics = {f"epoch_{key}": item for key, item in epoch_metrics.items()}
        return loss, epoch_metrics

    def training_epoch_end(self, training_step_outputs):
        training_step_outputs = [l["loss"] for l in training_step_outputs]
        loss, metrics = self._epoch_end(
            training_step_outputs, self.train_metrics.compute()
        )
        self.train_metrics.reset()
        self.log("epoch_train/loss", loss)
        self.log_dict(metrics)

    def validation_epoch_end(self, validation_step_outputs):
        loss, metrics = self._epoch_end(
            validation_step_outputs, self.valid_metrics.compute()
        )
        self.valid_metrics.reset()
        self.log("epoch_val/loss", loss)
        self.log_dict(metrics)

    def test_epoch_end(self, test_step_outputs):
        loss, metrics = self._epoch_end(test_step_outputs, self.test_metrics.compute())
        self.test_metrics.reset()
        self.log("epoch_test/loss", loss)
        self.log_dict(metrics)

    def configure_optimizers(self):
        return torch.optim.RAdam(self.parameters(), lr=self.lr)
