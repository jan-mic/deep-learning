import os
from datetime import datetime
import lightning as pl
import torch
from matplotlib import pyplot as plt
from torch import nn
from torchmetrics import F1Score
from torchmetrics.classification import BinaryAccuracy, BinaryConfusionMatrix


class CustomModel(pl.LightningModule):
    def __init__(self, n_blocks=3, dropout=0.5, learning_rate=1e-3, pos_weight=None):
        super().__init__()
        self.save_hyperparameters()
        self.register_buffer("pos_weight", torch.tensor(pos_weight) if pos_weight is not None else None)
        # inpput: 224x224x3 output: 112x112x32
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        # input: 112x112x32 output: 56x56x64
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        # input: 56x56x64 output: 28x28x128
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 1)
        )

        # define metrics
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

        self.train_acc = BinaryAccuracy()
        self.train_f1 = F1Score(task="binary")
        self.val_acc = BinaryAccuracy()
        self.val_f1 = F1Score(task="binary")
        self.test_acc = BinaryAccuracy()
        self.test_f1 = F1Score(task="binary")

        # self.pr_curve = PrecisionRecallCurve(task="binary")
        self.test_confmat = BinaryConfusionMatrix()

    def forward(self, x):
        # define forward pass
        for block in [self.block1, self.block2, self.block3]:
            x = block(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.float().view(-1, 1)
        y_hat = self(x)

        loss = self.criterion(y_hat, y.float())


        self.train_acc.update(y_hat.sigmoid(), y.int())
        self.train_f1.update(y_hat.sigmoid(), y.int())
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, on_epoch=True, on_step=False, prog_bar=True)
        self.log("train_f1", self.train_f1, on_epoch=True, on_step=False, prog_bar=True)


        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.float().view(-1, 1)
        y_hat = self(x)

        loss = self.criterion(y_hat, y.float())
        self.val_acc.update(y_hat.sigmoid(), y.int())
        self.val_f1.update(y_hat.sigmoid(), y.int())

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
        self.log("val_f1", self.val_f1, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.float().view(-1, 1)
        y_hat = self(x)

        probs = torch.sigmoid(y_hat)
        self.test_acc.update(probs, y.int())
        self.test_f1.update(probs, y.int())

        self.test_confmat.update(probs, y.int())

        # self.pr_curve.update(probs, y.int())
        self.log("test_acc", self.test_acc, prog_bar=True)
        self.log("test_f1", self.test_f1, prog_bar=True)

    def on_test_epoch_end(self):
        os.makedirs("charts", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"charts/confusion_matrix_{timestamp}.png"

        cm = self.test_confmat.compute()
        fig, ax = self.test_confmat.plot(cmap='Blues')  # TorchMetrics built-in plotting

        # If using TensorBoard, log the figure directly
        plt.savefig(filename)
        plt.close(fig)

        print(f"Confusion matrix saved to: {filename}")
        self.test_confmat.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        return {"optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss",
            }
        }