# pragma: no cover
# pylint: disable=too-many-locals, no-self-use

from typing import Optional

import attrs
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning import seed_everything

from zetta_utils import builder, distributions, tensor_ops, viz


@builder.register("EncodingCoarsenerGenX1Regime")
@attrs.mutable(eq=False)
class EncodingCoarsenerGenX1Regime(pl.LightningModule):  # pylint: disable=too-many-ancestors
    encoder: torch.nn.Module
    decoder: torch.nn.Module
    lr: float
    train_log_row_interval: int = 200
    val_log_row_interval: int = 25
    field_magn_thr: float = 1
    zero_value: float = 0
    worst_val_loss: float = attrs.field(init=False, default=0)
    worst_val_sample: dict = attrs.field(init=False, factory=dict)
    worst_val_sample_idx: Optional[int] = attrs.field(init=False, default=None)

    min_data_thr: float = 0.85
    equivar_weight: float = 1.0
    significance_weight: float = 0.5
    centering_weight: float = 0.5
    equivar_rot_deg_distr: distributions.Distribution = distributions.uniform_distr(0, 360)
    equivar_shear_deg_distr: distributions.Distribution = distributions.uniform_distr(-10, 10)
    equivar_trans_px_distr: distributions.Distribution = distributions.uniform_distr(-10, 10)
    equivar_scale_distr: distributions.Distribution = distributions.uniform_distr(0.9, 1.1)

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

    def validation_epoch_start(self, _):
        seed_everything(42)

    def validation_epoch_end(self, _):
        self.log_results(
            "val",
            "worst",
            **self.worst_val_sample,
        )
        self.worst_val_loss = 0
        self.worst_val_sample = {}
        self.worst_val_sample_idx = None
        seed_everything(None)

    def training_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
        log_row = batch_idx % self.train_log_row_interval == 0
        loss = self.compute_gen_x1_loss(batch=batch, mode="train", log_row=log_row)
        return loss

    def compute_gen_x1_loss(self, batch: dict, mode: str, log_row: bool, sample_name: str = ""):
        src = batch["src"]
        seed_field = batch["field"]
        seed_field = (
            seed_field * self.field_magn_thr / torch.quantile(seed_field.abs().max(1)[0], 0.5)
        ).field()
        if ((src == self.zero_value)).bool().sum() / src.numel() > self.min_data_thr:
            return None
        equivar_rot = self.equivar_rot_deg_distr()

        equivar_field = (
            tensor_ops.transform.get_affine_field(
                size=src.shape[-1],
                rot_deg=equivar_rot,
            )
            .field()
            .to(src.device)(seed_field.from_pixels())
            .pixels()
        )

        equivar_field_inv = (~seed_field.from_pixels())(
            tensor_ops.transform.get_affine_field(
                size=src.shape[-1],
                rot_deg=-equivar_rot,
            )
            .field()
            .to(src.device)
        ).pixels()

        src_warped = equivar_field.from_pixels()(src)
        enc_warped = self.encoder(src_warped)
        enc = tensor_ops.interpolate(
            equivar_field_inv, scale_factor=enc_warped.shape[-1] / src.shape[-1], mode="field"
        ).from_pixels()(enc_warped)
        dec = self.decoder(enc)

        tissue_final = equivar_field_inv.from_pixels()(
            equivar_field.from_pixels()(torch.ones_like(src))
        )

        diff_map = (src - dec).abs()
        diff_loss = diff_map[tissue_final != 0].sum()
        diff_map[tissue_final == 0] = 0
        wanted_significance = torch.nn.functional.max_pool2d(
            src.abs(), kernel_size=int(src.shape[-1] / enc.shape[-1])
        )
        enc_error = torch.nn.functional.avg_pool2d(
            diff_map.abs(), kernel_size=int(src.shape[-1] / enc.shape[-1])
        )
        significance_loss_map = (enc_error - (wanted_significance - enc.abs().mean(1))).abs()
        tissue_final_downs = torch.nn.functional.max_pool2d(
            tissue_final, kernel_size=int(src.shape[-1] / enc.shape[-1])
        )
        significance_loss = significance_loss_map[..., tissue_final_downs.squeeze() == 1].sum()
        significance_loss_map[..., tissue_final_downs.squeeze() == 0] = 0
        centering_loss = enc.sum((0, 2, 3)).abs().sum()
        loss = (
            diff_loss
            + self.significance_weight * significance_loss
            + self.centering_weight * centering_loss
        )
        self.log(f"loss/{mode}_significance", significance_loss, on_step=True, on_epoch=True)
        self.log(f"loss/{mode}_diff", diff_loss, on_step=True, on_epoch=True)
        self.log(f"loss/{mode}_centering", centering_loss, on_step=True, on_epoch=True)

        if log_row:
            self.log_results(
                mode,
                sample_name,
                src=src,
                src_warped=src_warped,
                src_warped_abs=src_warped.abs(),
                enc_warped_naive=tensor_ops.interpolate(
                    src_warped, size=(enc_warped.shape[-2], enc_warped.shape[-1]), mode="img"
                ),
                enc_warped=enc_warped,
                enc_warped_abs=enc_warped.abs().mean(1),
                enc=enc,
                dec=dec,
                diff_map=diff_map,
                significance_loss_map=significance_loss_map,
                enc_error=enc_error,
                waned_significance=wanted_significance,
                waned_zeros=wanted_significance == 0,
                enc_zeros=enc.abs().mean(1) < 0.01,
                # tissue_final_downs=tissue_final_downs,
            )

        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
        log_row = batch_idx % self.val_log_row_interval == 0
        sample_name = f"{batch_idx // self.val_log_row_interval}"

        loss = self.compute_gen_x1_loss(
            batch=batch, mode="val", log_row=log_row, sample_name=sample_name
        )
        return loss
