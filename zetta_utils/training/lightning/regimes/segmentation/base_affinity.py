# pragma: no cover
# pylint: disable=arguments-differ,too-many-ancestors

from __future__ import annotations

import attrs
import pytorch_lightning as pl
import torch

from zetta_utils import builder
from zetta_utils.segmentation import AffinityLoss

from ..common import log_results


@builder.register("BaseAffinityRegime")
@attrs.mutable(eq=False)
class BaseAffinityRegime(pl.LightningModule):
    model: torch.nn.Module
    criterion: AffinityLoss
    lr: float
    amsgrad: bool = True
    logits: bool = True

    train_log_row_interval: int = 200
    val_log_row_interval: int = 25

    def __attrs_pre_init__(self):
        super().__init__()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            amsgrad=self.amsgrad,
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        log_row = batch_idx % self.train_log_row_interval == 0
        loss = self.compute_loss(batch=batch, mode="train", log_row=log_row)
        return loss

    def validation_step(self, batch, batch_idx):
        log_row = batch_idx % self.val_log_row_interval == 0
        loss = self.compute_loss(batch=batch, mode="val", log_row=log_row)
        return loss

    def compute_loss(
        self, batch: dict[str, torch.Tensor], mode: str, log_row: bool, sample_name: str = ""
    ):
        data_in = batch["data_in"]
        target = batch["target"]
        if "target_mask" in batch:
            mask = batch["target_mask"]
        else:
            mask = torch.ones_like(target)

        result = self.model(data_in)
        loss = self.criterion(result, target, mask)
        if loss is None:
            return None

        self.log(f"loss/{mode}", loss.item(), on_step=True, on_epoch=True)

        if log_row:
            log_results(
                mode,
                sample_name,
                data_in=data_in,
                target=target,
                result=torch.sigmoid(result) if self.logits else result,
            )
        return loss
