# pragma: no cover
# pylint: disable=too-many-locals

from typing import Optional

import attrs
import cc3d
import pytorch_lightning as pl
import torch
import wandb

from zetta_utils import builder, distributions, tensor_ops, viz


@builder.register("BaseEncoderRegime")
@attrs.mutable(eq=False)
class BaseEncoderRegime(pl.LightningModule):  # pylint: disable=too-many-ancestors
    model: torch.nn.Module
    lr: float
    train_log_row_interval: int = 200
    val_log_row_interval: int = 25
    field_magn_thr: float = 1
    max_displacement_px: float = 16.0
    post_weight: float = 0.5
    zero_value: float = 0
    zero_conserve_weight: float = 0.5
    worst_val_loss: float = attrs.field(init=False, default=0)
    worst_val_sample: dict = attrs.field(init=False, factory=dict)
    worst_val_sample_idx: Optional[int] = attrs.field(init=False, default=None)

    equivar_weight: float = 1.0
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
        loss = self.compute_metroem_loss(batch=batch, mode="train", log_row=log_row)
        return loss

    def _get_warped(self, img, field):
        img_padded = torch.nn.functional.pad(img, (1,1,1,1), value=self.zero_value)  # TanH! - fill with output zero value
        img_warped = field.from_pixels()(img)

        zeros_padded = img_padded == self.zero_value
        zeros_padded_cc = cc3d.connected_components(zeros_padded.detach().squeeze().cpu().numpy(), connectivity=4).reshape(zeros_padded.shape)
        zeros_padded[torch.tensor(zeros_padded_cc != zeros_padded_cc.ravel()[0], device=zeros_padded.device)] = False  # keep masking resin, restore most soma

        zeros_warped = torch.nn.functional.pad(field, (1,1,1,1), mode='replicate').from_pixels().sample((~zeros_padded).float(), padding_mode='border') <= 0.1
        zeros_warped = torch.nn.functional.pad(zeros_warped, (-1,-1,-1,-1), value=True)

        img_warped[zeros_warped] = self.zero_value
        return img_warped, zeros_warped

    def compute_metroem_loss(self, batch: dict, mode: str, log_row: bool, sample_name: str = ""):
        src = batch["images"]["src"]
        tgt = batch["images"]["tgt"]

        if ((src == self.zero_value) + (tgt == self.zero_value)).bool().sum() / src.numel() > 0.4:
            return None

        seed_field = batch["field"].field_()
        f_warp_large = seed_field * self.max_displacement_px
        f_warp_small = (
            seed_field * self.field_magn_thr / torch.quantile(seed_field.abs().max(1)[0], 0.5)
        )

        f_aff = (
            tensor_ops.transform.get_affine_field(
                size=src.shape[-1],
                rot_deg=self.equivar_rot_deg_distr(),
                scale=self.equivar_scale_distr(),
                shear_x_deg=self.equivar_shear_deg_distr(),
                shear_y_deg=self.equivar_shear_deg_distr(),
                trans_x_px=self.equivar_trans_px_distr(),
                trans_y_px=self.equivar_trans_px_distr(),
            )
            .pixels()
            .to(seed_field.device)
        )
        f1_trans = f_aff.from_pixels()(f_warp_large.from_pixels()).pixels()
        f2_trans = f_warp_small.from_pixels()(f1_trans.from_pixels()).pixels()

        src_f1, src_zeros_f1 = self._get_warped(src, f1_trans)
        src_f2, src_zeros_f2 = self._get_warped(src, f2_trans)
        tgt_f1, tgt_zeros_f1 = self._get_warped(tgt, f1_trans)

        src_enc = self.model(src)
        src_f1_enc = self.model(src_f1)

        src_enc_f1 = torch.nn.functional.pad(src_enc, (1,1,1,1), value=0.0)
        src_enc_f1 = torch.nn.functional.pad(f1_trans, (1,1,1,1), mode='replicate').from_pixels().sample(src_enc_f1, padding_mode='border')  # type: ignore
        src_enc_f1 = torch.nn.functional.pad(src_enc_f1, (-1,-1,-1,-1), value=0.0)

        equi_diff = (src_enc_f1 - src_f1_enc).abs()
        # equi_loss = equi_diff[src_zeros_f1 == 0].sum()
        equi_loss = equi_diff.sum()
        equi_diff_map = equi_diff.clone()
        # equi_diff_map[src_zeros_f1] = 0

        src_f2_enc = self.model(src_f2)
        tgt_f1_enc = self.model(tgt_f1)

        pre_diff = (src_f1_enc - tgt_f1_enc).abs()

        pre_tissue_mask = tensor_ops.mask.coarsen(tgt_zeros_f1 + src_zeros_f1, width=2) == 0
        pre_loss = pre_diff[..., pre_tissue_mask].sum()
        pre_diff_masked = pre_diff.clone()
        pre_diff_masked[..., pre_tissue_mask == 0] = 0

        post_tissue_mask = tensor_ops.mask.coarsen(tgt_zeros_f1 + src_zeros_f2, width=2) == 0

        post_magn_mask = (f_warp_small.abs().max(1)[0] > self.field_magn_thr).tensor_()

        post_diff_map = (src_f2_enc - tgt_f1_enc).abs()
        post_mask = post_magn_mask * post_tissue_mask
        if post_mask.sum() < 256:
            return None

        post_loss = post_diff_map[..., post_mask].sum()

        post_diff_masked = post_diff_map.clone()
        post_diff_masked[..., post_mask == 0] = 0

        loss = pre_loss - post_loss * self.post_weight + equi_loss * self.equivar_weight
        self.log(f"loss/{mode}", loss, on_step=True, on_epoch=True)
        self.log(f"loss/{mode}_pre", pre_loss, on_step=True, on_epoch=True)
        self.log(f"loss/{mode}_post", post_loss, on_step=True, on_epoch=True)
        self.log(f"loss/{mode}_equi", equi_loss, on_step=True, on_epoch=True)
        if log_row:
            self.log_results(
                mode,
                sample_name,
                src=src,
                src_enc=src_enc,
                src_f1=src_f1,
                src_enc_f1=src_enc_f1,
                src_f1_enc=src_f1_enc,
                src_f2_enc=src_f2_enc,
                tgt_f1=tgt_f1,
                tgt_f1_enc=tgt_f1_enc,
                field=f_warp_small.tensor_(),
                equi_diff_map=equi_diff_map,
                post_diff_masked=post_diff_masked,
                pre_diff_masked=pre_diff_masked,
            )
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
        log_row = batch_idx % self.val_log_row_interval == 0
        sample_name = f"{batch_idx // self.val_log_row_interval}"

        loss = self.compute_metroem_loss(
            batch=batch, mode="val", log_row=log_row, sample_name=sample_name
        )
        return loss
