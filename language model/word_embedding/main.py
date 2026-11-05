import os
from datetime import datetime
from pathlib import Path

import lightning as pl
import pandas as pd
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from matplotlib import pyplot as plt

from task2.word_embedding.model import TadeuszTransformerModule, TransformerTadeusz
from task2.word_embedding.word_dataset import TadeuszBPEDataModule


def generate_charts(file_path="metrics.csv", version=None):
    metrics = pd.read_csv(file_path)

    train_loss = metrics[['epoch', 'train_loss']].dropna().groupby('epoch').mean()

    val_loss = metrics[['epoch', 'val_loss']].dropna().set_index('epoch')

    plt.figure(figsize=(10, 6))
    plt.plot(train_loss.index, train_loss['train_loss'], label='Average Train Loss', marker='o')
    plt.plot(val_loss.index, val_loss['val_loss'], label='Validation Loss', marker='s')

    plt.title('Training vs Validation Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    charts_dir = Path(__file__).parent / "charts" / "lstm"
    charts_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(charts_dir / f"loss_curve_{timestamp}_v{version}.png")


if __name__ == "__main__":
    data = TadeuszBPEDataModule()
    data.setup()
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    model = TadeuszTransformerModule(
        model=TransformerTadeusz(
            vocab_size=data.vocab_size, d_model=128, nhead=4, num_layers=8, dim_feedforward=512, dropout=0.2
        ),
        lr=1e-4
    )
    csv_logger = CSVLogger(save_dir="lightning_logs", name="memory_csv_logs")
    checkpoint_loss = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        filename="best-loss-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
        verbose=True,
        min_delta=0.002,
    )

    trainer = pl.Trainer(
        max_epochs=150,
        accelerator="auto",
        devices=1,
        log_every_n_steps=20,
        callbacks=[checkpoint_loss, early_stop_callback, lr_monitor],
        logger=csv_logger,
    )
    trainer.fit(model, datamodule=data)

    trainer.test(model, datamodule=data, ckpt_path="best")

    if os.path.exists(f"{csv_logger.log_dir}/metrics.csv"):
        generate_charts(file_path=f"{csv_logger.log_dir}/metrics.csv", version=csv_logger.version)
    else:
        print("No csv file found")
