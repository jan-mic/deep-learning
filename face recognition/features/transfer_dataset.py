from pathlib import Path
import lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CelebA
from torchvision.transforms import v2

CELEBA_DIR = Path(__file__).parent.parent / "data"
ATTR_MAP = {"Eyeglasses": 15, "Smiling": 31, "Male": 20}


def feature_name_to_idx(name):
    return ATTR_MAP.get(name, 15)


train_tfms = v2.Compose([
    v2.Pad(padding=(20, 0), fill=0),
    v2.Resize((224, 224)),
    v2.RandomHorizontalFlip(),
    v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    v2.RandomRotation(degrees=15),
    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_tfms = v2.Compose([
    v2.Pad(padding=(20, 0), fill=0),
    v2.Resize((224, 224)),
    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class CelebADataset(Dataset):
    def __init__(self, split, feature_idx=None, feature_name=None, transform=None):
        self.transform = transform
        self.samples = CelebA(root=CELEBA_DIR, split=split, target_type="attr", download=False,
                              transform=self.transform)
        self.attr = self.samples.attr

        if feature_name:
            self.feature_idx = feature_name_to_idx(feature_name)
        elif feature_idx is not None:
            self.feature_idx = int(feature_idx)
        else:
            self.feature_idx = 15

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        return img, label[self.feature_idx]


class CelebADataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=4, feature_idx=None, feature_name=None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        if feature_name:
            self.feature_idx = feature_name_to_idx(feature_name)
        elif feature_idx is not None:
            self.feature_idx = int(feature_idx)
        else:
            self.feature_idx = 15

    def setup(self, stage=None):
        self.train_ds = CelebADataset("train", self.feature_idx, transform=train_tfms)
        self.val_ds = CelebADataset("valid", self.feature_idx, transform=val_tfms)
        self.test_ds = CelebADataset("test", self.feature_idx, transform=val_tfms)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)
