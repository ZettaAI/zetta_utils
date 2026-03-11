from __future__ import annotations

import warnings

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
import torchvision
from torchvision import models, transforms

from zetta_utils import builder

from .cifar10 import CIFAR10_MEAN, CIFAR10_STD

warnings.filterwarnings("ignore", "Detected call of `lr_scheduler.step")


@builder.register("CIFAR10ResizedDataset")
class CIFAR10ResizedDataset(torch.utils.data.Dataset):  # pragma: no cover
    def __init__(self, train: bool = True, data_dir: str = "/tmp/cifar10", size: int = 224):
        if train:
            transform = transforms.Compose(
                [
                    transforms.Resize(size),
                    transforms.RandomCrop(size, padding=size // 8),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize(size),
                    transforms.ToTensor(),
                    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
                ]
            )
        self._dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=train, download=True, transform=transform
        )

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._dataset[idx]


@builder.register("ResNet50CIFAR10Regime")
class ResNet50CIFAR10Regime(pl.LightningModule):  # pragma: no cover
    def __init__(
        self,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_steps: int = 5000,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_steps = max_steps
        self.model = models.resnet50(num_classes=10)

    def forward(self, x):  # pylint: disable=arguments-differ
        return self.model(x)

    def training_step(self, batch, batch_idx):  # pylint: disable=arguments-differ,unused-argument
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, on_step=True)
        self.log("train_acc", acc, on_step=True)
        return loss

    def validation_step(
        self, batch, batch_idx
    ):  # pylint: disable=arguments-differ,unused-argument
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, on_epoch=True, sync_dist=True)
        self.log("val_acc", acc, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        scaled_lr = self.lr * (self.trainer.world_size ** 0.5) * 0.25
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=scaled_lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_steps)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
