from __future__ import annotations

from torch.utils.data import DataLoader

from task1.features.custom_dataset import CelebADataset, val_tfms, WiderFaceTestDataset

"""custom_analysis.py

Prosty, "ręczny" skrypt do debugowania predykcji.

Cel:
- wczytać wytrenowany model z checkpointu Lightning (.ckpt)
- przejść po N obrazach z CelebA (split=test)
- zapisać obrazki (matplotlib) z podpisami: true label / predicted label / probability
- osobno zapisać przykłady błędnie sklasyfikowane, żeby łatwo je przejrzeć

Uwaga: ten plik celowo NIE używa argparse/CLI — konfiguracja jest stałymi niżej.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.transforms.functional import to_pil_image

from custom_dataset import CelebADataModuleCustom
from custom_model import CustomModel

from captum.attr import GuidedBackprop
from captum.attr._utils.visualization import visualize_image_attr

WIDERFACE_DIR = Path(__file__).parent.parent / "data" / "widerface"
IMG_DIR = WIDERFACE_DIR / "transformed_padding"
CSV_PATH = WIDERFACE_DIR / "labels.csv"
# =====================
# KONFIGURACJA (EDYTUJ TU)
# =====================
BASE_DIR = Path(__file__).parent
FEATURE_NAME = "Male"  # np. "Eyeglasses", "Smiling"

# Najlepszy checkpoint (np. best-loss-...ckpt). Podmień na własną ścieżkę.
# Ważne: ścieżka jest liczona względem katalogu tego pliku (task1/features),
# więc skrypt będzie działał niezależnie od katalogu, z którego go uruchamiasz.
CKPT_PATH = BASE_DIR / "lightning_logs" / "tensorboard_logs" / "version_29" / "checkpoints" / "best-loss-epoch=15-val_loss=0.05.ckpt"

# Ile obrazów przeanalizować (domyślnie 20)
NUM_IMAGES = 5000

# Próg decyzyjny dla klasy pozytywnej
THRESHOLD = 0.5

# Czy zapisywać wizualizację atrybucji (GuidedBackprop)
SAVE_ATTRIBUTIONS = False

# Gdzie zapisać wyniki
OUT_DIR = Path(__file__).parent / "analysis" / FEATURE_NAME
SAVE_DPI = 160

# Ustawienia DataLoader
BATCH_SIZE = 1  # zostaw 1 dla prostoty i łatwego debugowania
NUM_WORKERS = 0  # 0 = najprościej na Windows (bez problemów z multiprocessingiem)

# Wymuszamy CPU, żeby było przewidywalnie
DEVICE = torch.device("cpu")

# Normalizacja taka sama jak w custom_dataset.py (val_tfms)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]


def _ensure_dirs() -> tuple[Path, Path]:
    correct_dir = OUT_DIR / "correct"
    wrong_dir = OUT_DIR / "wrong"
    correct_dir.mkdir(parents=True, exist_ok=True)
    wrong_dir.mkdir(parents=True, exist_ok=True)
    return correct_dir, wrong_dir


def _load_model() -> CustomModel:
    if not CKPT_PATH.exists():
        hint_dir = (BASE_DIR / "lightning_logs").resolve()
        raise FileNotFoundError(
            "Checkpoint not found:\n"
            f"  {CKPT_PATH.resolve()}\n\n"
            "Podpowiedź: sprawdź katalog z logami Lightning, np.:\n"
            f"  {hint_dir}"
        )

    model = CustomModel.load_from_checkpoint(CKPT_PATH, map_location=DEVICE)
    model.eval()
    model.freeze()
    model.to(DEVICE)
    return model


def _label_from_prob(prob: float) -> int:
    return 1 if prob >= THRESHOLD else 0


def _denormalize_img(img_tensor_chw: torch.Tensor) -> torch.Tensor:
    """Cofnięcie ImageNet Normalize dla wizualizacji.

    Wejście: [C,H,W] po Normalize.
    Wyjście: [C,H,W] w [0,1].
    """
    mean = IMAGENET_MEAN.to(dtype=img_tensor_chw.dtype, device=img_tensor_chw.device)
    std = IMAGENET_STD.to(dtype=img_tensor_chw.dtype, device=img_tensor_chw.device)
    return (img_tensor_chw * std + mean).clamp(0, 1)


def _to_pil_for_plot(img_tensor_chw: torch.Tensor):
    img_vis = _denormalize_img(img_tensor_chw.detach().cpu())
    return to_pil_image(img_vis)


def _normalize_attr_map(attr_2d: torch.Tensor) -> torch.Tensor:
    """Normalizuje mapę atrybucji do [0,1] (bezpiecznie dla płaskich map)."""
    a = attr_2d.detach().float().cpu()
    a = a - a.min()
    denom = a.max()
    if float(denom.item()) < 1e-12:
        return torch.zeros_like(a)
    return a / denom


def _compute_guided_backprop_attr(*, model: torch.nn.Module, x_bchw: torch.Tensor) -> torch.Tensor:
    """Liczy atrybucje wejścia metodą GuidedBackprop.

    Zwraca tensor atrybucji w formacie [C,H,W] (dla pojedynczego obrazu).

    Uwaga: wymagane gradienty; nie wołać w `torch.no_grad()`.
    """
    if x_bchw.ndim != 4 or x_bchw.shape[0] != 1:
        raise ValueError(f"Expected x shape [1,C,H,W], got {tuple(x_bchw.shape)}")

    x = x_bchw.detach().clone()
    x.requires_grad_(True)

    model.zero_grad(set_to_none=True)

    gbp = GuidedBackprop(model)
    attr = gbp.attribute(x, target=0)  # [1,C,H,W]
    return attr[0]


def _save_example_figure(
    *,
    img_tensor: torch.Tensor,
    true_label: int,
    pred_label: int,
    prob: float,
    out_path: Path,
    attr_chw: torch.Tensor | None = None,
    img_name: str | None = None,
) -> None:
    """Zapisuje jedną figurę.

    Jeśli attr_chw != None, zapisuje 2 panele: oryginał + wizualizacja atrybucji.

    Do wizualizacji atrybucji używa Captum: `captum.attr._utils.visualization.visualize_image_attr`.
    """

    pil = _to_pil_for_plot(img_tensor)

    name_part = f" | {img_name}" if img_name else ""

    if attr_chw is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(pil)
        ax.axis("off")
        fig.suptitle(f"{FEATURE_NAME}{name_part} | true={true_label} pred={pred_label} p={prob:.3f}")
        fig.tight_layout()
        fig.savefig(out_path, dpi=SAVE_DPI)
        plt.close(fig)
        return

    # Captum oczekuje obrazka jako numpy w [H,W,C]. Dajemy zdenormalizowany [0,1].
    img_hwc = _denormalize_img(img_tensor).permute(1, 2, 0).detach().cpu().numpy()

    # Atrybucje: [C,H,W] -> [H,W,C] jako numpy
    attr_hwc = attr_chw.detach().cpu().permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(img_hwc)
    axes[0].axis("off")
    axes[0].set_title("Original")

    visualize_image_attr(
        attr=attr_hwc,
        original_image=img_hwc,
        method="heat_map",
        sign="all",
        plt_fig_axis=(fig, axes[1]),
        title="GuidedBackprop",
        use_pyplot=False,
    )
    axes[1].axis("off")

    fig.suptitle(f"{FEATURE_NAME}{name_part} | true={true_label} pred={pred_label} p={prob:.3f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=SAVE_DPI)
    plt.close(fig)


def main() -> None:
    correct_dir, wrong_dir = _ensure_dirs()
    model = _load_model()


    loader = DataLoader(CelebADataset(split="test", feature_name=FEATURE_NAME, transform=val_tfms, return_filename=True), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    # loader = DataLoader(
    #     WiderFaceTestDataset(CSV_PATH, IMG_DIR, target_attribute=FEATURE_NAME),
    #     batch_size=BATCH_SIZE,
    #     shuffle=False,
    #     num_workers=NUM_WORKERS,
    # )
    processed = 0
    wrong = 0

    print(f"Saving to: {OUT_DIR}")
    if SAVE_ATTRIBUTIONS:
        print("Captum attributions: ON")
    print()

    for _, batch in enumerate(loader):
        if processed >= NUM_IMAGES:
            break

        # Obsługujemy oba formaty datasetu:
        # - (x, y)
        # - (x, y, filename)
        filename = None
        if isinstance(batch, (tuple, list)) and len(batch) == 3:
            x, y, filename = batch
            # filename z DataLoadera przy batch_size=1 jest listą/tuplem długości 1
            if isinstance(filename, (list, tuple)) and len(filename) == 1:
                filename = filename[0]
        else:
            x, y = batch

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        # Predykcja: bez gradientów
        with torch.no_grad():
            logits = model(x)  # [B,1]
            probs = torch.sigmoid(logits).view(-1)  # [B]

        prob = float(probs[0].item())
        true_label = int(y.view(-1)[0].item())
        pred_label = _label_from_prob(prob)

        is_correct = pred_label == true_label
        dst_dir = correct_dir if is_correct else wrong_dir
        if not is_correct:
            wrong += 1

        attr = None
        if SAVE_ATTRIBUTIONS:
            # atrybucja wymaga gradientów
            with torch.enable_grad():
                attr = _compute_guided_backprop_attr(model=model, x_bchw=x)

        safe_name = Path(str(filename)).name if filename else None
        out_path = dst_dir / f"{processed:04d}_true{true_label}_pred{pred_label}_p{prob:.3f}.png"
        if not is_correct:
            _save_example_figure(
                img_tensor=x[0],
                true_label=true_label,
                pred_label=pred_label,
                prob=prob,
                out_path=out_path,
                attr_chw=attr,
                img_name=safe_name,
            )

        processed += 1

    print(f"Done. Processed: {processed}")
    print(f"Wrong predictions: {wrong}")


if __name__ == "__main__":
    main()

