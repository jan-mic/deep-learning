from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from torchvision.datasets import CelebA

from celeba_attributes import CELEBA_ATTRIBUTES


# Trzymamy spójnie z resztą projektu
CELEBA_DIR = Path(__file__).parent.parent / "data"


@dataclass(frozen=True)
class SplitStats:
    split: str
    total: int
    pos: torch.Tensor  # shape: [num_attrs]


def _load_split_attrs(split: str) -> torch.Tensor:
    """Zwraca tensor atrybutów CelebA dla zadanego splitu.

    Torchvision CelebA zwraca attr jako tensor int (0/1) shape [N, 40].
    Nie wczytujemy obrazów do pamięci – pobieramy tylko macierz etykiet.
    """
    ds = CelebA(
        root=CELEBA_DIR,
        split=split,
        target_type="attr",
        download=True,
        transform=None,
    )

    # W torchvision to jest już torch.Tensor; trzymamy to defensywnie.
    attr = ds.attr
    if not isinstance(attr, torch.Tensor):
        attr = torch.as_tensor(attr)

    # Oczekujemy [N, 40].
    if attr.ndim != 2 or attr.shape[1] != len(CELEBA_ATTRIBUTES):
        raise ValueError(
            f"Unexpected attr shape for split={split}: {tuple(attr.shape)}; expected [N, {len(CELEBA_ATTRIBUTES)}]"
        )

    return attr


def _compute_split_stats(split: str) -> SplitStats:
    attr = _load_split_attrs(split)
    total = int(attr.shape[0])
    # attr jest 0/1, więc suma po wymiarze 0 daje liczbę pozytywów per cecha.
    pos = attr.sum(dim=0).to(torch.int64)
    return SplitStats(split=split, total=total, pos=pos)


def _stats_to_dataframe(splits: Iterable[SplitStats]) -> pd.DataFrame:
    """Buduje tabelę: wiersz = feature, kolumny = pos/neg/% dla splitów + total."""

    splits = list(splits)
    if not splits:
        raise ValueError("No splits provided")

    num_attrs = len(CELEBA_ATTRIBUTES)

    def _empty_col_dict(prefix: str) -> dict[str, list[float | int]]:
        return {
            f"{prefix}_pos": [0] * num_attrs,
            f"{prefix}_neg": [0] * num_attrs,
            f"{prefix}_pct": [0.0] * num_attrs,
        }

    # Start
    data: dict[str, list[float | int | str]] = {
        "feature_idx": list(range(num_attrs)),
        "feature": list(CELEBA_ATTRIBUTES),
    }

    # Per split columns
    for s in splits:
        cols = _empty_col_dict(s.split)
        pos_list = s.pos.tolist()
        neg_list = [(s.total - int(p)) for p in pos_list]
        pct_list = [100.0 * (float(p) / float(s.total)) if s.total else 0.0 for p in pos_list]

        cols[f"{s.split}_pos"] = [int(x) for x in pos_list]
        cols[f"{s.split}_neg"] = [int(x) for x in neg_list]
        cols[f"{s.split}_pct"] = pct_list
        data.update(cols)

    # Total across splits
    total_n = int(sum(s.total for s in splits))
    total_pos = torch.stack([s.pos for s in splits], dim=0).sum(dim=0)
    total_pos_list = total_pos.tolist()
    total_neg_list = [(total_n - int(p)) for p in total_pos_list]
    total_pct_list = [100.0 * (float(p) / float(total_n)) if total_n else 0.0 for p in total_pos_list]

    data["all_total"] = [total_n] * num_attrs
    data["all_pos"] = [int(x) for x in total_pos_list]
    data["all_neg"] = [int(x) for x in total_neg_list]
    data["all_pct"] = total_pct_list

    df = pd.DataFrame(data)

    # Czytelne sortowanie: najczęstsze cechy na górze
    df = df.sort_values(by="all_pct", ascending=False, kind="mergesort").reset_index(drop=True)
    return df


def analyze_celeba_attributes(save_csv: bool = True) -> pd.DataFrame:
    splits = [
        _compute_split_stats("train"),
        # _compute_split_stats("valid"),
        # _compute_split_stats("test"),
    ]

    df = _stats_to_dataframe(splits)

    # Print (skimmable)
    total_n = int(df.loc[0, "all_total"]) if len(df) else 0
    print("--- CelebA attribute distribution ---")
    print(f"Root: {CELEBA_DIR}")
    print(f"Total images (train+valid+test): {total_n}")
    print()

    # Zwięzła tabela w konsoli
    view = df[["feature", "all_pos", "all_pct"]].copy()
    view["all_pct"] = view["all_pct"].map(lambda x: f"{x:6.2f}%")
    print(view.to_string(index=False))

    if save_csv:
        out_path = Path(__file__).parent / "celeba_attribute_stats.csv"
        df.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path}")

    return df


if __name__ == "__main__":
    analyze_celeba_attributes(save_csv=True)

