from __future__ import annotations

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision import transforms

from zetta_utils import builder

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


@builder.register("CIFAR10Dataset")
class CIFAR10Dataset(torch.utils.data.Dataset):  # pragma: no cover
    def __init__(self, train: bool = True, data_dir: str = "/tmp/cifar10"):
        if train:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
                ]
            )
        else:
            transform = transforms.Compose(
                [
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


@builder.register("CIFAR10Regime")
class CIFAR10Regime(pl.LightningModule):  # pragma: no cover
    def __init__(self, lr: float = 1e-3, weight_decay: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay

        self.features = nn.Sequential(
            # Block 1: 3 -> 64
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2: 64 -> 128
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3: 128 -> 256
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 10),
        )

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

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
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
