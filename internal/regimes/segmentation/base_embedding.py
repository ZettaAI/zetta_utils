# pragma: no cover
# pylint: disable=arguments-differ,too-many-ancestors

from __future__ import annotations

import attrs
import pytorch_lightning as pl
import torch

from zetta_utils import builder, tensor_ops
from zetta_utils.training.lightning.train import distributed_available

from ...segmentation import vec_to_pca, vec_to_rgb
from ..common import log_3d_results


@builder.register("BaseEmbeddingRegime")
@attrs.mutable(eq=False)
class BaseEmbeddingRegime(pl.LightningModule):
    model: torch.nn.Module
    lr: float
    criteria: dict[str, torch.nn.Module]
    loss_weights: dict[str, float]
    amsgrad: bool = True

    train_log_row_interval: int = 200
    val_log_row_interval: int = 25

    # DDP
    sync_dist: bool = True

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
            pred = results[key]
            trgt = batch[key]
            mask = batch[key + "_mask"]
            splt = batch.get(key + "_split", None)
            loss = criterion(pred, trgt, mask, splt)
            if loss is None:
                continue
            loss_w = self.loss_weights[key]
            losses += [loss_w * loss]
            self.log(
                f"loss/{key}/{mode}",
                loss.item(),
                on_step=True,
                on_epoch=True,
                sync_dist=(distributed_available() and self.sync_dist),
                rank_zero_only=True,
            )

        if len(losses) == 0:
            return None

        loss = sum(losses)
        self.log(
            f"loss/{mode}",
            loss.item(),
            on_step=True,
            on_epoch=True,
            sync_dist=(distributed_available() and self.sync_dist),
            rank_zero_only=True,
        )

        if log_row:
            log_3d_results(
                self.logger,
                mode,
                title_suffix=sample_name,
                max_dims=None,
                **self.create_row(batch, results),
            )

        return loss

    def create_row(
        self, batch: dict[str, torch.Tensor], results: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        row = {
            "data_in": batch["data_in"],
            "target": tensor_ops.seg_to_rgb(batch["target"]),
        }

        for key in self.criteria.keys():
            mask = batch[key + "_mask"]
            pred = results[key]

            # PCA dimensionality reduction
            vec = pred[[0], ...]
            pca = vec_to_pca(vec)
            row[f"{key}[0:3]"] = vec_to_rgb(vec)
            row[f"{key}_PCA"] = vec_to_rgb(pca)

            # Optional mask
            if torch.count_nonzero(mask) < torch.numel(mask):
                row[f"{key}_mask"] = mask

        return row
