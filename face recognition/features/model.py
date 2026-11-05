import torch
import torch.nn as nn
import lightning as pl
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC


class EfficientNetB0Attribute(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        pos_weight: float | None = None,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.freeze_backbone = freeze_backbone
        # ---- Backbone ----
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        self.model = efficientnet_b0(weights=weights)

        # Replace classifier (binary output)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, 1)

        # Optional freezing (stage 1 transfer learning)
        if self.freeze_backbone:
            for p in self.model.features.parameters():
                p.requires_grad = False

        # ---- Loss ----
        if pos_weight is not None:
            self.criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(pos_weight)
            )
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        # ---- Metrics ----
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.test_acc = BinaryAccuracy()

        self.val_auc = BinaryAUROC()
        self.test_auc = BinaryAUROC()

    # ---------------------------------------------------
    # Forward
    # ---------------------------------------------------
    def forward(self, x):
        return self.model(x).squeeze(1)  # (B,)

    # ---------------------------------------------------
    # Training step
    # ---------------------------------------------------
    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()

        logits = self(x)
        loss = self.criterion(logits, y)

        probs = torch.sigmoid(logits)
        self.train_acc.update(probs, y.int())

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", self.train_acc, prog_bar=True)

        return loss

    # ---------------------------------------------------
    # Validation step
    # ---------------------------------------------------
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()

        logits = self(x)
        loss = self.criterion(logits, y)

        probs = torch.sigmoid(logits)
        self.val_acc.update(probs, y.int())
        self.val_auc.update(probs, y.int())

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", self.val_acc, prog_bar=True)
        self.log("val/auc", self.val_auc, prog_bar=True)

    # ---------------------------------------------------
    # Test step
    # ---------------------------------------------------
    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()

        logits = self(x)
        loss = self.criterion(logits, y)

        probs = torch.sigmoid(logits)
        self.test_acc.update(probs, y.int())
        self.test_auc.update(probs, y.int())

        self.log("test/loss", loss)
        self.log("test/acc", self.test_acc)
        self.log("test/auc", self.test_auc)

    # ---------------------------------------------------
    # Optimizer
    # ---------------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=3,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/auc",
            },
        }