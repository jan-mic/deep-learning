import os
from pathlib import Path
from datetime import datetime

import pandas as pd
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from custom_dataset import CelebADataModuleCustom
from custom_model import CustomModel
from custom_dataset import WiderFaceTestDataset

NUM_WORKERS = 8

CELEBA_DIR = Path(__file__).parent.parent / "data"
torch.set_float32_matmul_precision('medium')
FEATURE_NAME = "Male"

WIDERFACE_DIR = Path(__file__).parent.parent / "data" / "widerface"
IMG_DIR = WIDERFACE_DIR / "transformed_padding"
CSV_PATH = WIDERFACE_DIR / "labels.csv"

def generate_charts(file_path = "metrics.csv", version=None):
    # Load the metrics
    metrics = pd.read_csv(file_path)

    # 1. Process Training Loss: Group by epoch and take the mean
    # This handles multiple step-based logs in your CSV (e.g., steps 49, 99, 149 in Epoch 0)
    train_loss = metrics[['epoch', 'train_loss']].dropna().groupby('epoch').mean()

    # 2. Process Validation Loss: Drop NaNs (it is logged once per epoch)
    val_loss = metrics[['epoch', 'val_loss']].dropna().set_index('epoch')

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss.index, train_loss['train_loss'], label='Average Train Loss', marker='o')
    plt.plot(val_loss.index, val_loss['val_loss'], label='Validation Loss', marker='s')

    plt.title('Training vs Validation Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save to charts directory
    charts_dir = Path(__file__).parent / "charts" / FEATURE_NAME
    charts_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(charts_dir / f"loss_curve_{timestamp}_v{version}.png")
    # plt.show()


def export_best_checkpoint_to_onnx(best_ckpt_path: str, export_dir: Path, feature_name: str, version: int | str):
    """Eksportuje najlepszy checkpoint Lightning (.ckpt) do ONNX.

    Kontrakt:
    - wejście: tensor float32 NCHW (B, 3, 224, 224)
    - wyjście: logity (B, 1)

    Uwaga: wymuszamy CPU (model + dummy input), żeby uniknąć błędów typu
    "Tensor on device cuda:0 is not on the expected device cpu!".

    Uwaga 2: eksportujemy bezpośrednio do opset 18, bo konwerter 18->17 potrafi się wywalić
    (np. na adapterze axes_input_to_attribute).
    """
    export_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = export_dir / f"{feature_name}_v{version}_best.onnx"

    device = torch.device("cpu")

    best_model = CustomModel.load_from_checkpoint(best_ckpt_path, map_location=device)
    best_model.eval()
    best_model.freeze()
    best_model.to(device)
    dummy = torch.randn(1, 3, 224, 224, dtype=torch.float32, device=device)

    # Utrzymuj opset >= 18, żeby nie uruchamiać konwersji w dół.
    opset_version = 18

    with torch.no_grad():
        torch.onnx.export(
            best_model,
            (dummy,),
            onnx_path.as_posix(),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        )

    return onnx_path


if __name__ == "__main__":


    data = CelebADataModuleCustom(
        batch_size=32,
        num_workers=NUM_WORKERS,
        feature_name=FEATURE_NAME,
    )
    data.setup()

    model = CustomModel(
        learning_rate=1e-3
    )

    tb_logger = TensorBoardLogger(save_dir="lightning_logs", name="tensorboard_logs")

    csv_logger = CSVLogger(save_dir="lightning_logs", name="csv_logs")

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    checkpoint = ModelCheckpoint(
        monitor="val_f1",
        mode="max",
        filename="best-model-{epoch:02d}-{val_f1:.2f}",
        save_top_k=1
    )

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
        min_delta=0.001,
        stopping_threshold=0.001
    )

    trainer = Trainer(
        max_epochs=30,
        accelerator="gpu",
        devices=1,
        log_every_n_steps=100,
        logger=[tb_logger, csv_logger],
        callbacks=[early_stop_callback, checkpoint, checkpoint_loss, lr_monitor],
        # check_val_every_n_epoch=1,
    )

    trainer.fit(model, data)

    best_ckpt = checkpoint_loss.best_model_path

    # Test na najlepszym checkpointcie (a nie na ostatnim modelu w pamięci)
    trainer.test(model=model, datamodule=data, ckpt_path=best_ckpt)


    if os.path.exists(f"{csv_logger.log_dir}/metrics.csv"):
        generate_charts(file_path=f"{csv_logger.log_dir}/metrics.csv", version=csv_logger.version)
    else:
        print("No csv file found")

    test_loader = DataLoader(
        WiderFaceTestDataset(CSV_PATH, IMG_DIR, target_attribute=FEATURE_NAME),
        batch_size=32,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    trainer.test(model=model, dataloaders=test_loader, ckpt_path=best_ckpt)
    # Eksport BEST ONNX do final_models
    export_dir = Path(__file__).parent / "final_models"
    onnx_path = export_best_checkpoint_to_onnx(
        best_ckpt_path=best_ckpt,
        export_dir=export_dir,
        feature_name=FEATURE_NAME,
        version=csv_logger.version,
    )
    print(f"ONNX zapisany do: {onnx_path}")
