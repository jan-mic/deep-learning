from pathlib import Path

import torch
from lightning import Trainer
from torchvision.datasets import CelebA
from torchvision.models import EfficientNet_B0_Weights

from model import EfficientNetB0Attribute
from task1.features.dataset import CelebAAttributeDataset, CelebADataModule

CELEBA_DIR = Path(__file__).parent.parent / "data"
torch.set_float32_matmul_precision('medium')



if __name__ == '__main__':
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    train_tfms = weights.transforms()
    val_tfms = weights.transforms()

    train = CelebA(root=CELEBA_DIR, split="train", download=True, transform=train_tfms)


    train_attr_data = CelebAAttributeDataset(split="train", feature_idx=15)
    val_attr_data = CelebAAttributeDataset(split="valid", feature_idx=15)
    test_attr_data = CelebAAttributeDataset(split="test", feature_idx=15)
    print(
        f"Train: {len(train_attr_data)}, Val: {len(val_attr_data)}, Test: {len(test_attr_data)}"
    )
    train_loader = torch.utils.data.DataLoader(train_attr_data, batch_size=32, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(val_attr_data, batch_size=32, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
    test_loader = torch.utils.data.DataLoader(test_attr_data, batch_size=32, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

    data = CelebADataModule(batch_size=32, num_workers=8, feature_idx=15)
    # data.prepare_data()
    # data.setup()
    # print(
    #     f"Train: {len(data.train_ds)}, Val: {len(data.val_ds)}, Test: {len(data.test_ds)}"
    # )
    model = EfficientNetB0Attribute(
        lr=1e-4,
        freeze_backbone=True,  # stage 1
    )

    trainer = Trainer(
        max_epochs=10,
        accelerator="auto",
        devices=1,
        log_every_n_steps=20,
    )

    trainer.fit(model, datamodule=data)
    model.freeze_backbone = False
    model.hparams.lr = 1e-5
    # model = EfficientNetB0Attribute(
    #     lr=1e-5,
    #     freeze_backbone=False,
    # )
    #
    trainer = Trainer(
        max_epochs=5,
        accelerator="auto",
        devices=1,
        log_every_n_steps=20,
    )
    trainer.fit(model, datamodule=data)

    trainer.test(model, datamodule=data)