# pylint: disable=arguments-differ,no-self-use,too-many-ancestors
import os
from typing import Literal, Sequence

import attrs
import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from torch.nn import BCELoss, BCEWithLogitsLoss

from zetta_utils import builder
from zetta_utils.internal.segmentation import BinaryClassBalancer
from zetta_utils.internal.segmentation.loss import LossWithMask
from zetta_utils.tensor_ops import crop

from .common import log_3d_results, log_results


@builder.register("BinarySupervisedRegime")
@attrs.mutable(eq=False)
class BinarySupervisedRegime(pl.LightningModule):
    model: torch.nn.Module
    lr: float
    optimizer: torch.optim.Optimizer | None = None

    loss: torch.nn.Module | None = None
    logits: bool = False
    loss_crop_pad: Sequence[int] | None = None

    reduction_mode: Literal["sum", "mean"] = "mean"
    mean_reduction_clip: float = 0.1

    mask_loss_on_zeros: bool = False
    min_nonz_frac: float = 0.0
    min_nonz_frac_on_mask: bool = False

    class_balance_mode: Literal["default", "frequency", "none"] = "none"
    class_balance_clip: float = 0.1

    data_in_key: str = "data_in_img"
    target_key: str = "target_seg"

    train_log_row_interval: int = 200
    val_log_row_interval: int = 25
    log_max_dims: Sequence[int] | None = None

    """
    :param mean_reduction_clip:
        Clip multiplier when computing mean loss when loss mask is partial, i.e.,
        multipler = sum / max(mask_ratio, clip).

    :param mask_loss_on_zeros:
        Remove mask on pixels where img == 0.

    :param min_nonz_frac:
        Skip batches with non zero image ratio less than this.
    .. note: NOT compatible with DDP training.

    :param min_nonz_frac_on_mask:
        Make min_nonz_frac applied to the mask instead of images.

    :class_balance_mode:
        Choose mode for class balancing.
    .. note: "default" & "frequency" are dynamically computed within the batch.
        "none" disables balancing. TODO: add static balancing

    :class_balance_clip:
        Clip ratios for class balancing.
    """

    def __attrs_pre_init__(self):
        super().__init__()

    def __attrs_post_init__(self):
        seed_everything(42, workers=True)

        if self.loss is None:
            balancer = None
            if self.class_balance_mode != "none":
                balancer = BinaryClassBalancer(
                    group=1,
                    clipmin=self.class_balance_clip,
                    clipmax=1 - self.class_balance_clip,
                    dynamic_mode=self.class_balance_mode,
                )
            loss = LossWithMask(
                criterion=BCEWithLogitsLoss if self.logits else BCELoss,
                reduction=self.reduction_mode,
                balancer=balancer,
                mean_reduction_clip=self.mean_reduction_clip,
                return_loss_map=True,
            )
            object.__setattr__(self, "loss", loss)

        ddp_is_used = int(os.getenv("WORLD_SIZE", "1")) > 1
        if ddp_is_used and self.min_nonz_frac > 0:
            raise RuntimeError("min_nonz_frac is not supported with DDP training!")

    def configure_optimizers(self):
        if self.optimizer is not None:
            return self.optimizer(self.parameters(), lr=self.lr)  # type: ignore
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    # def on_validation_epoch_start(self):
    #     seed_everything(42)

    # def on_validation_epoch_end(self):
    #     seed_everything(None)

    def validation_step(self, batch, batch_idx):
        log_row = batch_idx % self.val_log_row_interval == 0
        sample_name = f"{batch_idx // self.val_log_row_interval}"
        loss = self.compute_loss(
            batch=batch,
            mode="val",
            log_row=log_row,
            sample_name=sample_name,
        )
        return loss

    def compute_loss(  # pylint: disable=too-many-branches
        self,
        batch: dict,
        mode: str,
        log_row: bool,
        sample_name: str = "",
    ):
        data_in = batch[self.data_in_key].to(torch.float32)
        target = batch[self.target_key] != 0
        target = target.to(torch.float32)
        for i in [-1, -2, -3]:
            if data_in.shape[i] != 1 and data_in.shape[i] % 16 != 0:
                raise ValueError(f"Data in shape {data_in.shape} is not divisible by 16")

        if self.loss_crop_pad is not None:
            data_in_cropped = crop(data_in, self.loss_crop_pad)
        else:
            data_in_cropped = data_in
        if (data_in_cropped != 0).sum() / data_in_cropped.numel() < self.min_nonz_frac:
            return None

        mask_key = self.target_key + "_mask"
        if mask_key in batch:
            target_mask = batch[mask_key].to(torch.float32)
        else:
            target_mask = torch.ones_like(target, dtype=torch.float32)

        if self.min_nonz_frac > 0:
            min_data_in = data_in
            if self.min_nonz_frac_on_mask:
                min_data_in = target_mask
            if self.loss_crop_pad is not None:
                min_data_in = crop(min_data_in, self.loss_crop_pad)
            if (min_data_in != 0).sum() / min_data_in.numel() < self.min_nonz_frac:
                return None

        prediction = self.model(data_in)

        if self.loss_crop_pad is not None:
            # prediction = crop(prediction, self.loss_crop_pad)
            # target = crop(target, self.loss_crop_pad)
            # target_mask = crop(target_mask, self.loss_crop_pad)
            assert not any(e == 0 for e in self.loss_crop_pad)
            target_mask[..., : self.loss_crop_pad[0], :, :] = 0
            target_mask[..., -self.loss_crop_pad[0] :, :, :] = 0

            target_mask[..., :, : self.loss_crop_pad[1], :] = 0
            target_mask[..., :, -self.loss_crop_pad[1] :, :] = 0

            target_mask[..., :, :, : self.loss_crop_pad[2]] = 0
            target_mask[..., :, :, -self.loss_crop_pad[2] :] = 0

        if self.mask_loss_on_zeros:
            target_mask[data_in == 0] = 0

        assert self.loss is not None
        loss, loss_map = self.loss(prediction, target, target_mask)

        self.log(f"loss/{mode}", loss.item(), on_step=(mode == "train"), on_epoch=True)

        if log_row:
            if self.logits:
                prediction = torch.sigmoid(prediction) if self.logits else prediction
            if data_in.shape[-1] != 1 and len(data_in.shape) > 4:
                log_3d_results(
                    logger=self.logger,
                    mode=mode,
                    title_suffix=sample_name,
                    data_in=data_in,
                    target=target,
                    prediction=prediction.to(torch.float32),
                    mask=target_mask,
                    max_dims=self.log_max_dims,
                )
            else:
                # To make sure mask has full range of values
                mask = target_mask[0].clone()
                mask[..., -1, -1] = True
                mask[..., 0, 0] = False
                target_vis = target[0].clone()
                target_vis[..., -1, -1] = True
                target_vis[..., 0, 0] = False
                log_results(
                    logger=self.logger,
                    mode=mode,
                    title_suffix=sample_name,
                    data_in=data_in[0],
                    target=target_vis,
                    prediction=prediction[0],
                    loss_map=loss_map[0],
                    mask=mask,
                )
        return loss

    def training_step(self, batch, batch_idx):

        # clear cache once every while to reduce gpu mem usage
        if batch_idx % 100 == 0:
            torch.cuda.empty_cache()

        log_row = batch_idx % self.train_log_row_interval == 0
        sample_name = ""

        loss = self.compute_loss(
            batch=batch,
            mode="train",
            log_row=log_row,
            sample_name=sample_name,
        )
        return loss
