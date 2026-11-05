from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import lightning as pl


def split_to_chars(text, keep_whitespace=True):
    text = text.lower()
    if keep_whitespace:
        arr = np.fromiter(text, dtype="<U1")
    else:
        joined = "".join(text.split())
        if not joined:
            return [], []
        arr = np.fromiter(joined, dtype="<U1")

    tokens = arr.tolist()
    unique_chars = np.unique(arr).tolist()
    return tokens, unique_chars


def char_to_index_dict(unique_chars):
    return {ch: i for i, ch in enumerate(unique_chars)}


def index_to_char_dict(unique_chars):
    return {i: ch for i, ch in enumerate(unique_chars)}


class TadeuszDataset(Dataset):
    def __init__(self, text, seq_length, char2idx, step=1):
        self.seq_length = int(seq_length)
        mapper = np.vectorize(char2idx.__getitem__, otypes=[np.int64])
        self.data = mapper(text)
        self.indices = np.arange(0, len(self.data) - self.seq_length - 1, step)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_length]
        y = self.data[idx + 1: idx + self.seq_length + 1]

        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class TadeuszDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, seq_length=150, step=1, num_workers=8):
        super().__init__()
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.step = step
        self.char2idx = None
        self.idx2char = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.vocab_size = None
        self.num_workers = num_workers

    def setup(self, stage=None):
        base_dir = Path(__file__).resolve().parent
        data_path = (base_dir / ".." / "data" / "pantadeusz_clean.txt").resolve()

        if data_path.exists():
            text = data_path.read_text(encoding="utf-8")
            parsed, uniq = split_to_chars(text)
            self.vocab_size = len(uniq)
            self.char2idx = char_to_index_dict(uniq)
            self.idx2char = index_to_char_dict(uniq)
            train, test, val = np.split(np.array(parsed), [int(0.8 * len(parsed)), int(0.9 * len(parsed))])
            self.train_ds = TadeuszDataset(train, self.seq_length, self.char2idx, self.step)
            self.val_ds = TadeuszDataset(val, self.seq_length, self.char2idx, self.step)
            self.test_ds = TadeuszDataset(test, self.seq_length, self.char2idx, self.step)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                          pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          pin_memory=True, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


if __name__ == "__main__":
    data = TadeuszDataModule()
    data.setup()
