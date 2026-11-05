import torch
import pandas as pd
from tqdm import tqdm
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from task1.features.transfer_model2 import TransferModel
from task1.features.dataset import CelebADataModule

def calculate_auto_pos_weight(datamodule):
    print("\n--- Automatyczne obliczanie pos_weight ---")
    datamodule.setup()
    temp_loader = datamodule.train_dataloader()

    num_pos = 0
    num_neg = 0

    print("Skanowanie zbioru treningowego...")
    for batch in tqdm(temp_loader, desc="Liczenie klas"):
        if isinstance(batch, (list, tuple)):
            labels = batch[1]
        else:
            labels = batch['label']

        num_pos += torch.sum(labels == 1).item()
        num_neg += torch.sum(labels == 0).item()

    if num_pos == 0:
        print("UWAGA: Nie znaleziono przykładów pozytywnych! Ustawiam wagę na 1.0")
        return 1.0

    weight = num_neg / num_pos
    print(f"Znaleziono: Pozytywne={num_pos}, Negatywne={num_neg}")
    print(f"Obliczona waga (pos_weight): {weight:.4f}")
    print("------------------------------------------\n")
    return weight

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    FEATURE_IDX = 15
    BATCH_SIZE = 32
    LOG_DIR = "csv_logs"

    data = CelebADataModule(batch_size=BATCH_SIZE, num_workers=8, feature_idx=FEATURE_IDX)
    calculated_weight = calculate_auto_pos_weight(data)
    model = TransferModel(lr=1e-3, pos_weight=calculated_weight)

    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="best_stage3")
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    early_stopping = EarlyStopping(monitor="val_loss", patience=3, mode="min", verbose=True)

    print(f"--- Stage 1: Head Only (pos_weight={calculated_weight:.2f}) ---")
    model.stage1_freeze_backbone()
    logger_stage1 = CSVLogger(LOG_DIR, name="Stage1_HeadOnly")

    trainer_stage1 = Trainer(
        max_epochs=10,
        accelerator="auto",
        devices=1,
        logger=logger_stage1,
        enable_checkpointing=False
    )
    trainer_stage1.fit(model, datamodule=data)

    print("--- Stage 2: Partial Fine-Tuning ---")
    model.stage2_partial_unfreeze()
    logger_stage2 = CSVLogger(LOG_DIR, name="Stage2_Partial")

    trainer_stage2 = Trainer(
        max_epochs=5,
        accelerator="auto",
        devices=1,
        logger=logger_stage2,
        enable_checkpointing=False
    )
    trainer_stage2.fit(model, datamodule=data)

    print("--- Stage 3: Full Fine-Tuning ---")
    model.stage3_unfreeze_all()
    logger_stage3 = CSVLogger(LOG_DIR, name="Stage3_Full")

    trainer_stage3 = Trainer(
        max_epochs=15,
        accelerator="auto",
        devices=1,
        logger=logger_stage3,
        callbacks=[checkpoint_callback, lr_monitor, early_stopping]
    )
    trainer_stage3.fit(model, datamodule=data)

    print("--- Testing ---")
    trainer_stage3.test(model, datamodule=data, ckpt_path="best")

    print(f"\nGotowe! Logi CSV znajdziesz w folderze: {LOG_DIR}")