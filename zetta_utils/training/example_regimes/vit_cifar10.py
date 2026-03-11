from __future__ import annotations

import warnings

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torchvision import models

from zetta_utils import builder

warnings.filterwarnings("ignore", "Detected call of `lr_scheduler.step")


@builder.register("ViTB16CIFAR10Regime")
class ViTB16CIFAR10Regime(pl.LightningModule):  # pragma: no cover
    def __init__(
        self,
        lr: float = 1e-3,
        weight_decay: float = 0.05,
        max_steps: int = 10000,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_steps = max_steps
        self.model = models.vit_b_16(num_classes=10)

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
