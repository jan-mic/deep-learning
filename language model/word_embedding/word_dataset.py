from pathlib import Path

import lightning as pl
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast


class TadeuszDatasetBPE(Dataset):
    def __init__(self, token_ids, seq_length, step=1):
        self.seq_length = int(seq_length)
        self.data = token_ids
        self.indices = np.arange(0, len(self.data) - self.seq_length - 1, step)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        x = self.data[idx: idx + self.seq_length]
        y = self.data[idx + 1: idx + self.seq_length + 1]

        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class TadeuszBPEDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, seq_length=150, step=1, num_workers=8):
        super().__init__()
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.step = step
        self.num_workers = num_workers
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file="tadeusz_tokenizer.json")

        self.vocab_size = self.tokenizer.vocab_size

    def setup(self, stage=None):
        data_files = [
            "../data/pantadeusz_clean.txt",
            "../data/quovadis_clean.txt",
            "../data/lalka_tom_pierwszy_clean.txt",
            "../data/lalka_tom_drugi_clean.txt",
            "../data/w_pustyni_i_w_puszczy_clean.txt",
        ]

        all_train_tokens = []
        all_val_tokens = []
        all_test_tokens = []

        for file_path in data_files:
            path = Path(file_path).resolve()
            if path.exists():
                text = path.read_text(encoding="utf-8").lower()

                encoded = self.tokenizer.encode(text, add_special_tokens=False)
                tokens = np.array(encoded)
                n = len(tokens)
                train_idx = int(0.9 * n)
                test_idx = int(0.95 * n)

                all_train_tokens.append(tokens[:train_idx])
                all_val_tokens.append(tokens[train_idx:test_idx])
                all_test_tokens.append(tokens[test_idx:])

        self.train_tokens = np.concatenate(all_train_tokens)
        self.val_tokens = np.concatenate(all_val_tokens)
        self.test_tokens = np.concatenate(all_test_tokens)

        self.train_ds = TadeuszDatasetBPE(self.train_tokens, self.seq_length, self.step)
        self.val_ds = TadeuszDatasetBPE(self.val_tokens, self.seq_length, self.step)
        self.test_ds = TadeuszDatasetBPE(self.test_tokens, self.seq_length, self.step)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, persistent_workers=True,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, persistent_workers=True,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers)


if __name__ == "__main__":
    data = TadeuszBPEDataModule()
    data.setup()
    test = '''Litwo! Ojczyzno moja! ty jesteś jak zdrowie:
    Ile cię trzeba cenić, ten tylko się dowie,
    Kto cię stracił. Dziś piękność twą w całej ozdobie
    Widzę i opisuję, bo tęsknię po tobie.'''
    encoded = data.tokenizer.encode(test, add_special_tokens=False)
    print(encoded)

    print(data.tokenizer.decode(encoded))
