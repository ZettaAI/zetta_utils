# pragma: no cover
# pylint: disable=arguments-differ,too-many-ancestors

from __future__ import annotations

from typing import NamedTuple

import attrs
import pytorch_lightning as pl
import torch

from zetta_utils import builder, tensor_ops

from ..common import log_3d_results


@builder.register("BaseAffinityRegime")
@attrs.mutable(eq=False)
class BaseAffinityRegime(pl.LightningModule):
    model: torch.nn.Module
    lr: float
    criteria: dict[str, torch.nn.Module]
    loss_weights: dict[str, float]
    amsgrad: bool = True
    logits: bool = True
    group: int = 3

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
        results = self.model(data_in)

        # Create mask if not exist
        for key, criterion in self.criteria.items():
            if key + "_mask" in batch:
                continue
            batch[key + "_mask"] = torch.ones_like(batch[key])

        # Compute loss
        losses = []
        for key, criterion in self.criteria.items():
            pred = getattr(results, key)
            trgt = batch[key]
            mask = batch[key + "_mask"]
            loss = criterion(pred, trgt, mask)
            if loss is None:
                continue
            loss_w = self.loss_weights[key]
            losses += [loss_w * loss]
            self.log(f"loss/{key}/{mode}", loss.item(), on_step=True, on_epoch=True)

        if len(losses) == 0:
            return None

        loss = sum(losses)
        self.log(f"loss/{mode}", loss.item(), on_step=True, on_epoch=True)

        if log_row:
            log_3d_results(
                mode,
                title_suffix=sample_name,
                **self.create_row(batch, results),
            )

        return loss

    def create_row(
        self, batch: dict[str, torch.Tensor], results: NamedTuple
    ) -> dict[str, torch.Tensor]:
        row = {
            "data_in": batch["data_in"],
            "target": tensor_ops.seg_to_rgb(batch["target"]),
        }

        for key in self.criteria.keys():
            trgt = batch[key]
            mask = batch[key + "_mask"]
            pred = getattr(results, key)
            pred = torch.sigmoid(pred) if self.logits else pred

            # Chop prediction into groups for visualization purpose
            num_channels = pred.shape[-4]
            group = self.group if self.group > 0 else num_channels
            if num_channels > group:
                for i in range(0, num_channels, group):
                    start, end = i, min(i + group, num_channels)
                    idx = f"[{start}:{end}]"
                    row[f"{key}{idx}"] = pred[..., start:end, :, :, :]
            else:
                row[f"{key}"] = pred

            # Chop target into groups for visualization purpose
            num_channels = trgt.shape[-4]
            group = self.group if self.group > 0 else num_channels
            if num_channels > group:
                for i in range(0, num_channels, group):
                    start, end = i, min(i + group, num_channels)
                    idx = f"[{start}:{end}]"
                    row[f"{key}_target{idx}"] = trgt[..., start:end, :, :, :]

                    # Optional mask
                    mask_ = mask[..., start:end, :, :, :]
                    if torch.count_nonzero(mask_) < torch.numel(mask_):
                        row[f"{key}_mask{idx}"] = mask_
            else:
                row[f"{key}_target"] = trgt

                # Optional mask
                if torch.count_nonzero(mask) < torch.numel(mask):
                    row[f"{key}_mask"] = mask

        return row
