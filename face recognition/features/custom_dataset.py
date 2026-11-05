from pathlib import Path

import lightning as pl
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.datasets import CelebA
from torchvision.transforms import v2

from celeba_attributes import feature_name_to_idx

CELEBA_DIR = Path(__file__).parent.parent / "data"

train_tfms = v2.Compose([
    v2.Pad(padding=(20, 0), fill=0),
    v2.Resize((224, 224)),
    v2.RandomHorizontalFlip(),
    v2.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.1,
        hue=0.05,
    ),
    v2.RandomRotation(
        degrees=15,
    ),
    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

val_tfms = v2.Compose([
    v2.Pad(padding=(20, 0), fill=0),
    v2.Resize((224, 224)),
    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class WiderFaceTestDataset(Dataset):
    def __init__(self, csv_file, img_dir, target_attribute):
        self.df = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)
        self.target_attribute = target_attribute

        # This transform must match your TRAINING transform exactly
        self.transform = val_tfms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['image_name']

        # Ensure label is float for BCEWithLogitsLoss
        label = float(row[self.target_attribute])

        img_path = self.img_dir / img_name
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor([label], dtype=torch.float32)

class CelebADataset(Dataset):
    def __init__(self, split, feature_idx=None, feature_name=None, transform=None, return_filename: bool = False):
        """CelebA dataset zwracający obraz oraz jedną wybraną cechę binarną.

        Możesz podać:
        - feature_idx (0..39) albo
        - feature_name (np. "Smiling")

        Jeśli podasz oba, priorytet ma feature_name.

        return_filename:
            Jeśli True, __getitem__ zwraca (img, label, filename).
            Domyślnie False, żeby nie zmieniać istniejącego kodu treningowego.
        """
        self.transform = transform
        self.return_filename = bool(return_filename)
        self.samples = CelebA(
            root=CELEBA_DIR,
            split=split,
            target_type="attr",
            download=True,
            transform=self.transform,
        )
        self.attr = self.samples.attr

        if feature_name is not None:
            self.feature_idx = feature_name_to_idx(feature_name)
        elif feature_idx is not None:
            self.feature_idx = int(feature_idx)
        else:
            raise ValueError("Pass feature_idx or feature_name")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        y = label[self.feature_idx]

        if not self.return_filename:
            return img, y

        # torchvision.datasets.CelebA exposes filenames via .filename
        filename = None
        if hasattr(self.samples, "filename"):
            filename = self.samples.filename[idx]
        return img, y, filename


class CelebADataModuleCustom(pl.LightningDataModule):
    def __init__(
            self,
            batch_size=32,
            num_workers=4,
            feature_idx=None,
            feature_name=None,
    ):
        super().__init__()
        self.test_ds = None
        self.val_ds = None
        self.train_ds = None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sampler = None
        if feature_name is not None:
            self.feature_idx = feature_name_to_idx(feature_name)
        elif feature_idx is not None:
            self.feature_idx = int(feature_idx)
        else:
            raise ValueError("Musisz podać feature_idx albo feature_name")

        self.train_tfms = train_tfms
        self.val_tfms = val_tfms

    def setup(self, stage=None):
        self.train_ds = CelebADataset(
            split="train",
            feature_idx=self.feature_idx,
            transform=self.train_tfms,
            return_filename=False,
        )
        labels = self.train_ds.attr[:, self.feature_idx]
        counts = labels.bincount()
        class_weights = 1.0 / counts.float()
        sample_weights = class_weights[labels]


        self.sampler = WeightedRandomSampler(
            sample_weights,
            len(self.train_ds),
            replacement=True
        )

        self.val_ds = CelebADataset(
            split="valid",
            feature_idx=self.feature_idx,
            transform=self.val_tfms
        )
        self.test_ds = CelebADataset(
            split="test",
            feature_idx=self.feature_idx,
            transform=self.val_tfms
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=False, # shuffle=False when using sampler
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            sampler=self.sampler
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
