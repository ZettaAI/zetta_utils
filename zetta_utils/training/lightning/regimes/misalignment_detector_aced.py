from typing import Optional

import attrs
import pytorch_lightning as pl
import torch
import wandb

from zetta_utils import builder, distributions, tensor_ops, viz


@builder.register("MisalignmentDetectorAcedRegime")
@attrs.mutable(eq=False)
class MisalignmnetDetectorAcedRegime(pl.LightningModule):  # pylint: disable=too-many-ancestors
    model: torch.nn.Module
    lr: float
    train_log_row_interval: int = 200
    val_log_row_interval: int = 25
    field_magn_thr: float = 1
    zero_value: float = 0

    worst_val_loss: float = attrs.field(init=False, default=0)
    worst_val_sample: dict = attrs.field(init=False, factory=dict)
    worst_val_sample_idx: Optional[int] = attrs.field(init=False, default=None)

    def __attrs_pre_init__(self):
        super().__init__()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    @staticmethod
    def log_results(mode: str, title_suffix: str = "", **kwargs):
        wandb.log(
            {
                f"results/{mode}_{title_suffix}_slider": [
                    wandb.Image(viz.rendering.Renderer()(v.squeeze()), caption=k)  # type: ignore
                    for k, v in kwargs.items()
                ]
            }
        )

    def validation_epoch_end(self, _):
        self.log_results(
            "val",
            "worst",
            **self.worst_val_sample,
        )
        self.worst_val_loss = 0
        self.worst_val_sample = {}
        self.worst_val_sample_idx = None

    def training_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
        log_row = batch_idx % self.train_log_row_interval == 0
        loss = self.compute_misd_loss(batch=batch, mode="train", log_row=log_row)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
        log_row = batch_idx % self.val_log_row_interval == 0
        sample_name = f"{batch_idx // self.val_log_row_interval}"

        loss = self.compute_misd_loss(
            batch=batch, mode="val", log_row=log_row, sample_name=sample_name
        )
        return loss

    def compute_misd_loss(self, batch: dict, mode: str, log_row: bool, sample_name: str = ""):
        src = batch["images"]["src"]
        tgt = batch["images"]["tgt"]

        if ((src == self.zero_value) + (tgt == self.zero_value)).bool().sum() / src.numel() > 0.7:
            return None

        src_field = (
            batch["field"]
            * self.field_magn_thr
            / torch.quantile(batch["field"].abs().max(1)[0], 0.5)
        )
        tgt_field = torch.zeros_like(src_field)
        """torch.tensor(
            src_field.field()
            .from_pixels()(src_field.field().from_pixels())
            .pixels()
        )
        """
        gt_labels = src_field.abs().max(1)[0] > self.field_magn_thr

        src_warped = src_field.field().from_pixels()(src)
        src_warped_tissue = src_field.field().from_pixels()((src != 0).float()) > 0.0
        tgt_warped = tgt_field.field().from_pixels()(tgt)  # type: ignore
        tgt_warped_tissue = tgt_field.field().from_pixels()((tgt != 0).float()) > 0.0  # type: ignore
        joint_tissue = tgt_warped_tissue + src_warped_tissue
        # breakpoint()
        prediction = self.model(torch.cat((src_warped, tgt_warped), 1))
        # loss_map = torch.nn.functional.binary_cross_entropy(
        # loss_map = torch.nn.functional.binary_cross_entropy_with_logits(
        #    prediction,
        #    gt_labels.unsqueeze(1).float(),
        #    reduction='none'
        # )
        loss_map = (prediction - gt_labels.unsqueeze(1).float()).abs()
        loss = loss_map[joint_tissue].sum()
        loss_map_masked = loss_map.clone()
        loss_map_masked[joint_tissue == 0] = 0

        self.log(f"loss/{mode}", loss, on_step=True, on_epoch=True)

        if log_row:
            self.log_results(
                mode,
                sample_name,
                src=src,
                tgt=tgt,
                src_warped=src_warped,
                tgt_warped=tgt_warped,
                src_warped_tissue=src_warped_tissue,
                tgt_warped_tissue=tgt_warped_tissue,
                joint_tissue=joint_tissue,
                gt_labels=gt_labels,
                prediction=prediction,
                loss_map=loss_map,
                loss_map_masked=loss_map_masked,
            )
        return loss
