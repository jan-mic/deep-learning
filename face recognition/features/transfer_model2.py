import os
from datetime import datetime
import lightning as pl
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryConfusionMatrix

class TransferModel(pl.LightningModule):
    def __init__(self, lr=1e-3, pos_weight=None):
        super().__init__()
        self.save_hyperparameters()

        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        for param in self.model.features.parameters():
            param.requires_grad = False

        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 1)

        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor([pos_weight]))
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        self.acc = BinaryAccuracy()
        self.f1 = BinaryF1Score()
        self.test_confmat = BinaryConfusionMatrix()
        self.learning_rate = lr

    def stage1_freeze_backbone(self):
        for param in self.model.features.parameters():
            param.requires_grad = False
        self.learning_rate = 1e-3

    def stage2_partial_unfreeze(self):
        for param in self.model.features[:6].parameters():
            param.requires_grad = False
        for param in self.model.features[6:].parameters():
            param.requires_grad = True
        self.learning_rate = 1e-4

    def stage3_unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = True
        self.learning_rate = 1e-5

    def forward(self, x):
        return self.model(x).squeeze(1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y.float())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y.float())
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        probs = torch.sigmoid(logits)
        self.acc(probs, y)
        self.f1(probs, y)
        self.log("test_acc", self.acc, prog_bar=True)
        self.log("test_f1", self.f1, prog_bar=True)
        self.test_confmat.update(probs, y.int())

    def on_test_epoch_end(self):
        os.makedirs("charts", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"charts/confusion_matrix_{timestamp}.png"

        fig, ax = self.test_confmat.plot(cmap='Blues')
        plt.savefig(filename)
        plt.close(fig)

        print(f"Confusion matrix saved to: {filename}")
        self.test_confmat.reset()

    def configure_optimizers(self):
        params_to_update = filter(lambda p: p.requires_grad, self.parameters())
        return torch.optim.Adam(params_to_update, lr=self.learning_rate)