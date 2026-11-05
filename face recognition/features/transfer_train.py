import torch
from pathlib import Path
from torch.utils.data import DataLoader
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from task1.features.transfer_model import TransferModel
from task1.features.transfer_dataset import CelebADataModule
from custom_dataset import WiderFaceTestDataset


def calculate_auto_pos_weight(datamodule):
    print("--- Obliczanie pos_weight ---")
    datamodule.setup()
    labels = datamodule.train_ds.attr[:, datamodule.feature_idx]

    num_pos = torch.sum(labels == 1).item()
    num_neg = torch.sum(labels == 0).item()

    if num_pos == 0: return 1.0
    weight = num_neg / num_pos
    print(f"Poz: {num_pos}, Neg: {num_neg}, Waga: {weight:.4f}")
    return weight


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    FEATURE_NAME = "Eyeglasses"
    BATCH_SIZE = 32
    LOG_DIR = "csv_logs"

    base_dir = Path(__file__).resolve().parent.parent / "data" / "widerface"
    wider_csv = base_dir / "labels.csv"
    wider_img_dir = base_dir / "transformed_padding"

    data = CelebADataModule(batch_size=BATCH_SIZE, num_workers=8, feature_name=FEATURE_NAME)
    pos_weight = calculate_auto_pos_weight(data)
    model = TransferModel(lr=1e-3, pos_weight=pos_weight)

    callbacks = [
        ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="best_stage2"),
        LearningRateMonitor(logging_interval='epoch'),
        EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=True)
    ]

    print("--- Stage 1 ---")
    model.stage1_freeze_backbone()
    Trainer(max_epochs=10, logger=CSVLogger(LOG_DIR, name="Stage1"), enable_checkpointing=False).fit(model,
                                                                                                    datamodule=data)

    print("--- Stage 2 ---")
    model.stage2_partial_unfreeze()
    trainer = Trainer(max_epochs=10, logger=CSVLogger(LOG_DIR, name="Stage2"), callbacks=callbacks)
    trainer.fit(model, datamodule=data)

    print("\n=== Test CelebA ===")
    trainer.test(model, datamodule=data, ckpt_path="best")

    print("\n=== Test WIDER Face ===")
    if wider_csv.exists() and wider_img_dir.exists():
        wider_ds = WiderFaceTestDataset(wider_csv, wider_img_dir, FEATURE_NAME)
        if len(wider_ds) > 0:
            loader = DataLoader(wider_ds, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)
            trainer.test(model, dataloaders=loader, ckpt_path="best")
    else:
        print(f"Pominięto WIDER Face (brak plików w {base_dir})")