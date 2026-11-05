from pathlib import Path
import torch
from torch.utils.data import DataLoader
from lightning import Trainer


from custom_model import CustomModel
from transfer_model import TransferModel
from custom_dataset import WiderFaceTestDataset, val_tfms


MODEL_TYPE = "transfer"


ATTRIBUTE_NAME = "Eyeglasses"

CHECKPOINT_PATH = "/home/jd/PycharmProjects/deep-learning/task1/features/lightning_logs/Transfer_Eyeglasses/version_5/checkpoints/best-epoch=00-val_loss=0.04.ckpt"


BASE_DIR = Path(__file__).parent.parent / "data" / "widerface"
CSV_PATH = BASE_DIR / "labels.csv"
IMG_DIR = BASE_DIR / "transformed_padding"


# ---------------------------------------------

def run_test():
    print(f"--- ROZPOCZYNAM TESTOWANIE NA WIDER FACE ---")
    print(f"Model: {MODEL_TYPE}")
    print(f"Atrybut: {ATTRIBUTE_NAME}")
    print(f"Plik CSV: {CSV_PATH}")


    if not CSV_PATH.exists():
        print(f"BŁĄD: Nie znaleziono pliku {CSV_PATH}")
        return


    test_dataset = WiderFaceTestDataset(
        csv_file=CSV_PATH,
        img_dir=IMG_DIR,
        target_attribute=ATTRIBUTE_NAME
    )

    if len(test_dataset) == 0:
        print("BŁĄD: Dataset jest pusty! Sprawdź labels.csv.")
        return

    print(f"Załadowano {len(test_dataset)} zdjęć do testów.")

    # DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        num_workers=4,
        shuffle=False
    )

    print(f"Ładowanie wag z: {CHECKPOINT_PATH}")

    if MODEL_TYPE == "custom":
        model = CustomModel.load_from_checkpoint(CHECKPOINT_PATH)
    elif MODEL_TYPE == "transfer":
        # TransferModel czasami wymagał podania lr w init, ale load_from_checkpoint to ogarnie
        model = TransferModel.load_from_checkpoint(CHECKPOINT_PATH)
    else:
        raise ValueError("Nieznany typ modelu! Wybierz 'custom' lub 'transfer'.")

    model.eval()
    model.freeze()


    trainer = Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False
    )


    trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    run_test()