"""Eksport checkpointu Lightning (.ckpt) do ONNX.

Użycie (Windows / PowerShell):
  python export_onnx.py --ckpt "C:\\path\\to\\model.ckpt" --feature Eyeglasses --version 0

Domyślnie zapisuje do: ./final_models/{feature}_v{version}_best.onnx

Uwagi:
- Skrypt zakłada taki sam kontrakt wejścia jak w treningu: (B, 3, 224, 224) float32.
- Jeśli trenowałeś model z innymi parametrami architektury/hparamami, muszą być zapisane w checkpointcie
  (Lightning zapisuje je jako hparams) – wtedy load_from_checkpoint odtworzy model poprawnie.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from custom_model import CustomModel


def export_ckpt_to_onnx(ckpt_path: Path, onnx_path: Path, device: str = "cpu") -> Path:
    ckpt_path = ckpt_path.expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Nie znaleziono checkpointu: {ckpt_path}")

    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    # podczas eksportu ONNX trzymaj wszystko na jednym urządzeniu
    # (na Windows najbezpieczniej: CPU)
    model = CustomModel.load_from_checkpoint(str(ckpt_path), map_location="cpu")
    model.eval()
    model.freeze()

    if device != "cpu":
        # umożliwiamy np. cuda, ale tylko jeśli użytkownik chce
        model = model.to(device)

    dummy = torch.randn(1, 3, 224, 224, dtype=torch.float32, device=device)

    torch.onnx.export(
        model,
        (dummy,),
        onnx_path.as_posix(),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        # wymusza starszy exporter, omija torch.export (który często powoduje problemy)
        dynamo=False,
    )

    return onnx_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Eksport Lightning .ckpt do ONNX")
    parser.add_argument("--ckpt", required=True, help="Ścieżka do pliku .ckpt")
    parser.add_argument(
        "--out",
        default=None,
        help="Ścieżka wyjściowa .onnx (opcjonalnie). Jeśli nie podasz, użyjemy ./final_models/<feature>_v<version>_best.onnx",
    )
    parser.add_argument("--feature", required=True, help="Nazwa cechy (np. Eyeglasses)")
    parser.add_argument("--version", required=True, help="Wersja CSVLogger (np. 0)")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Urządzenie dla eksportu (domyślnie cpu). Przykłady: cpu, cuda:0",
    )

    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)

    if args.out:
        onnx_path = Path(args.out)
    else:
        export_dir = Path(__file__).parent / "final_models"
        onnx_path = export_dir / f"{args.feature}_v{args.version}_best.onnx"

    out = export_ckpt_to_onnx(ckpt_path=ckpt_path, onnx_path=onnx_path, device=args.device)
    print(f"Zapisano ONNX: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
