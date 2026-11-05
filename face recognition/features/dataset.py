from pathlib import Path

import torch
import lightning as pl
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CelebA
from torchvision.transforms import v2


CELEBA_DIR = Path(__file__).parent.parent / "data"

train_tfms = v2.Compose([
    v2.Resize((224, 224)),  # no cropping
    # transforms.RandomHorizontalFlip(),    # faces are symmetric
    # transforms.ColorJitter(
    #     brightness=0.2,
    #     contrast=0.2,
    #     saturation=0.2,
    #     hue=0.05,
    # ),
    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),  # [0,1]
    v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

val_tfms = v2.Compose([
    v2.Resize((224, 224)),
    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class CelebAAttributeDataset(Dataset):
    def __init__(self, split, feature_idx):
        """
        samples: list of (image_path, label)
        label: 0 or 1
        """
        self.transform = train_tfms if split == "train" else val_tfms
        self.samples = CelebA(root=CELEBA_DIR, split=split, target_type="attr", download=True, transform=self.transform)
        self.feature_idx = feature_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]

        return img, label[self.feature_idx]




class CelebADataModule(pl.LightningDataModule):
    def __init__(
            self,
            batch_size=32,
            num_workers=4,
            feature_idx=15,
    ):
        super().__init__()
        self.test_ds = None
        self.val_ds = None
        self.train_ds = None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.feature_idx = feature_idx
        self.train_tfms = train_tfms
        self.val_tfms = val_tfms

    def setup(self, stage=None):
        self.train_ds = CelebAAttributeDataset(
            split="train",
            feature_idx=self.feature_idx,
        )
        self.val_ds = CelebAAttributeDataset(
            split="valid",
            feature_idx=self.feature_idx
        )
        self.test_ds = CelebAAttributeDataset(
            split="test",
            feature_idx=self.feature_idx
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
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
