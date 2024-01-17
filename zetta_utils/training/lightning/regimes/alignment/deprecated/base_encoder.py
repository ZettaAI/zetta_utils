# type: ignore
# pragma: no cover
# pylint: disable=too-many-locals, function-redefined

from typing import Optional

import attrs
import cc3d
import einops
import numpy as np
import pytorch_lightning as pl
import torch
import torchfields
import wandb
from PIL import Image as PILImage
from pytorch_lightning import seed_everything

from zetta_utils import builder, distributions, tensor_ops, viz
from zetta_utils.training.lightning.regimes.common import log_results


@builder.register("BaseEncoderRegime", versions="==0.0.1")
@attrs.mutable(eq=False)
class BaseEncoderRegime(pl.LightningModule):  # pylint: disable=too-many-ancestors
    model: torch.nn.Module
    lr: float
    train_log_row_interval: int = 200
    val_log_row_interval: int = 25
    field_magn_thr: float = 1
    max_displacement_px: float = 16.0
    post_weight: float = 1.5
    zero_value: float = 0
    worst_val_loss: float = attrs.field(init=False, default=0)
    worst_val_sample: dict = attrs.field(init=False, factory=dict)
    worst_val_sample_idx: Optional[int] = attrs.field(init=False, default=None)

    equivar_weight: float = 1.0
    equivar_rot_deg_distr: distributions.Distribution = distributions.uniform_distr(0, 360)
    equivar_shear_deg_distr: distributions.Distribution = distributions.uniform_distr(-10, 10)
    equivar_trans_px_distr: distributions.Distribution = distributions.uniform_distr(-10, 10)
    equivar_scale_distr: distributions.Distribution = distributions.uniform_distr(0.9, 1.1)
    empty_tissue_threshold: float = 0.4

    def __attrs_pre_init__(self):
        super().__init__()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def log_results(self, mode: str, title_suffix: str = "", **kwargs):
        if self.logger is None:
            return
        images = []
        for k, v in kwargs.items():
            for b in range(1):
                if v.dtype in (np.uint8, torch.uint8):
                    img = v[b].squeeze()
                    img[-1, -1] = 255
                    img[-2, -2] = 255
                    img[-1, -2] = 0
                    img[-2, -1] = 0
                    images.append(
                        wandb.Image(
                            PILImage.fromarray(viz.rendering.Renderer()(img), mode="RGB"),
                            caption=f"{k}_b{b}",
                        )
                    )
                elif v.dtype in (torch.int8, np.int8):
                    img = v[b].squeeze().byte() + 127
                    img[-1, -1] = 255
                    img[-2, -2] = 255
                    img[-1, -2] = 0
                    img[-2, -1] = 0
                    images.append(
                        wandb.Image(
                            PILImage.fromarray(viz.rendering.Renderer()(img), mode="RGB"),
                            caption=f"{k}_b{b}",
                        )
                    )
                elif v.dtype in (torch.bool, bool):
                    img = v[b].squeeze().byte() * 255
                    img[-1, -1] = 255
                    img[-2, -2] = 255
                    img[-1, -2] = 0
                    img[-2, -1] = 0
                    images.append(
                        wandb.Image(
                            PILImage.fromarray(viz.rendering.Renderer()(img), mode="RGB"),
                            caption=f"{k}_b{b}",
                        )
                    )
                else:
                    v_min = v[b].min().round(decimals=4)
                    v_max = v[b].max().round(decimals=4)
                    images.append(
                        wandb.Image(
                            viz.rendering.Renderer()(v[b].squeeze()),
                            caption=f"{k}_b{b} | min: {v_min} | max: {v_max}",
                        )
                    )

        self.logger.log_image(f"results/{mode}_{title_suffix}_slider", images=images)

    def validation_epoch_start(self, _):  # pylint: disable=no-self-use
        seed_everything(42)

    def on_validation_epoch_end(self):
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

        with torchfields.set_identity_mapping_cache(True, clear_cache=False):
            loss = self.compute_metroem_loss(batch=batch, mode="train", log_row=log_row)

        return loss

    def _get_warped(self, img, field=None):
        img_padded = torch.nn.functional.pad(img, (1, 1, 1, 1), value=self.zero_value)
        if field is not None:
            img_warped = field.from_pixels()(img)
        else:
            img_warped = img

        zeros_padded = img_padded == self.zero_value
        zeros_padded_cc = np.array(
            [
                cc3d.connected_components(
                    x.detach().squeeze().cpu().numpy(), connectivity=4
                ).reshape(zeros_padded[0].shape)
                for x in zeros_padded
            ]
        )

        non_tissue_zeros_padded = zeros_padded.clone()
        non_tissue_zeros_padded[
            torch.tensor(zeros_padded_cc != zeros_padded_cc.ravel()[0], device=zeros_padded.device)
        ] = False  # keep masking resin, restore somas in center

        if field is not None:
            zeros_warped = (
                torch.nn.functional.pad(field, (1, 1, 1, 1), mode="replicate")
                .from_pixels()  # type: ignore[attr-defined]
                .sample((~zeros_padded).float(), padding_mode="border")
                <= 0.1
            )
            non_tissue_zeros_warped = (
                torch.nn.functional.pad(field, (1, 1, 1, 1), mode="replicate")
                .from_pixels()  # type: ignore[attr-defined]
                .sample((~non_tissue_zeros_padded).float(), padding_mode="border")
                <= 0.1
            )
        else:
            zeros_warped = zeros_padded
            non_tissue_zeros_warped = non_tissue_zeros_padded

        zeros_warped = torch.nn.functional.pad(zeros_warped, (-1, -1, -1, -1))
        non_tissue_zeros_warped = torch.nn.functional.pad(
            non_tissue_zeros_warped, (-1, -1, -1, -1)
        )

        img_warped[zeros_warped] = self.zero_value
        return img_warped, ~zeros_warped, ~non_tissue_zeros_warped

    def compute_metroem_loss(self, batch: dict, mode: str, log_row: bool, sample_name: str = ""):
        src = batch["images"]["src_img"]
        tgt = batch["images"]["tgt_img"]

        if (
            (src == self.zero_value) + (tgt == self.zero_value)
        ).bool().sum() / src.numel() > self.empty_tissue_threshold:
            return None

        seed_field = batch["field"].field_()
        f_warp_large = seed_field * self.max_displacement_px
        f_warp_small = (
            seed_field * self.field_magn_thr / torch.quantile(seed_field.abs().max(1)[0], 0.5)
        )

        f_aff = (
            einops.rearrange(
                tensor_ops.transform.get_affine_field(
                    size=src.shape[-1],
                    rot_deg=self.equivar_rot_deg_distr(),
                    scale=self.equivar_scale_distr(),
                    shear_x_deg=self.equivar_shear_deg_distr(),
                    shear_y_deg=self.equivar_shear_deg_distr(),
                    trans_x_px=self.equivar_trans_px_distr(),
                    trans_y_px=self.equivar_trans_px_distr(),
                ),
                "C X Y Z -> Z C X Y",
            )
            .pixels()  # type: ignore[attr-defined]
            .to(seed_field.device)
        ).repeat_interleave(src.size(0), dim=0)
        f1_trans = f_aff.from_pixels()(f_warp_large.from_pixels()).pixels()
        f2_trans = f_warp_small.from_pixels()(f1_trans.from_pixels()).pixels()

        magn_field = f_warp_small

        src_f1, _, src_nonzeros_f1 = self._get_warped(src, f1_trans)
        src_f2, _, src_nonzeros_f2 = self._get_warped(src, f2_trans)
        tgt_f1, _, tgt_nonzeros_f1 = self._get_warped(tgt, f1_trans)

        src_zeros_f1 = ~src_nonzeros_f1
        src_zeros_f2 = ~src_nonzeros_f2
        tgt_zeros_f1 = ~tgt_nonzeros_f1

        src_enc = self.model(src)
        src_f1_enc = self.model(src_f1)

        src_enc_f1 = torch.nn.functional.pad(src_enc, (1, 1, 1, 1), value=0.0)
        src_enc_f1 = (
            torch.nn.functional.pad(f1_trans, (1, 1, 1, 1), mode="replicate")  # type: ignore
            .from_pixels()
            .sample(src_enc_f1, padding_mode="border")
        )
        src_enc_f1 = torch.nn.functional.pad(src_enc_f1, (-1, -1, -1, -1), value=0.0)

        equi_diff = (src_enc_f1 - src_f1_enc).abs()
        equi_loss = equi_diff[src_zeros_f1 != 0].sum()
        equi_loss = equi_diff.sum() / equi_diff.size(0)
        equi_diff_map = equi_diff.clone()
        equi_diff_map[src_zeros_f1] = 0

        src_f2_enc = self.model(src_f2)
        tgt_f1_enc = self.model(tgt_f1)

        pre_diff = (src_f1_enc - tgt_f1_enc).abs()

        pre_tissue_mask = ~(tgt_zeros_f1 | src_zeros_f1)
        pre_loss = pre_diff[..., pre_tissue_mask].sum() / pre_diff.size(0)
        pre_diff_masked = pre_diff.clone()
        pre_diff_masked[..., pre_tissue_mask == 0] = 0

        post_tissue_mask = ~(tgt_zeros_f1 | src_zeros_f2)
        post_magn_mask = (magn_field.abs().max(1, keepdim=True)[0] > self.field_magn_thr).tensor_()

        post_diff_map = (src_f2_enc - tgt_f1_enc).abs()
        post_mask = post_magn_mask * post_tissue_mask
        if post_mask.sum() < 256:
            return None

        post_loss = post_diff_map[..., post_mask].sum() / post_diff_map.size(0)

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
                field=seed_field.tensor_(),
                equi_diff_map=equi_diff_map,
                post_diff_masked=post_diff_masked,
                pre_diff_masked=pre_diff_masked,
            )
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
        log_row = batch_idx % self.val_log_row_interval == 0
        sample_name = f"{batch_idx // self.val_log_row_interval}"

        with torchfields.set_identity_mapping_cache(True, clear_cache=False):
            loss = self.compute_metroem_loss(
                batch=batch, mode="val", log_row=log_row, sample_name=sample_name
            )
        return loss


@builder.register("BaseEncoderRegime", versions="==0.0.0")
@attrs.mutable(eq=False)
class BaseEncoderRegime(pl.LightningModule):  # pylint: disable=too-many-ancestors
    model: torch.nn.Module
    lr: float
    train_log_row_interval: int = 200
    val_log_row_interval: int = 25
    field_magn_thr: float = 1
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

    def validation_epoch_end(self, _):
        log_results(
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
        img_warped = field.field().from_pixels()(img)
        zeros_warped = field.field().from_pixels()((img == self.zero_value).float()) > 0.1
        img_warped[zeros_warped] = 0
        return img_warped, zeros_warped

    def compute_metroem_loss(self, batch: dict, mode: str, log_row: bool, sample_name: str = ""):
        src = batch["images"]["src"]
        tgt = batch["images"]["tgt"]

        if ((src == self.zero_value) + (tgt == self.zero_value)).bool().sum() / src.numel() > 0.4:
            return None

        seed_field = batch["field"]
        seed_field = (
            seed_field * self.field_magn_thr / torch.quantile(seed_field.abs().max(1)[0], 0.5)
        )

        f_aff = (
            einops.rearrange(
                tensor_ops.transform.get_affine_field(
                    size=src.shape[-1],
                    rot_deg=self.equivar_rot_deg_distr(),
                    scale=self.equivar_scale_distr(),
                    shear_x_deg=self.equivar_shear_deg_distr(),
                    shear_y_deg=self.equivar_shear_deg_distr(),
                    trans_x_px=self.equivar_trans_px_distr(),
                    trans_y_px=self.equivar_trans_px_distr(),
                ),
                "C X Y Z -> Z C X Y",
            )
            .field()  # type: ignore
            .pixels()
            .to(seed_field.device)
        )
        f1_trans = torch.tensor(f_aff.from_pixels()(seed_field.field().from_pixels()).pixels())
        f2_trans = torch.tensor(
            seed_field.field()
            .from_pixels()(f1_trans.field().from_pixels())  # type: ignore
            .pixels()
        )

        src_f1, src_zeros_f1 = self._get_warped(src, f1_trans)
        src_f2, src_zeros_f2 = self._get_warped(src, f2_trans)
        tgt_f1, tgt_zeros_f1 = self._get_warped(tgt, f1_trans)

        src_enc = self.model(src)
        src_enc_f1 = f1_trans.field().from_pixels()(src_enc)  # type: ignore
        src_f1_enc = self.model(src_f1)

        equi_diff = (src_enc_f1 - src_f1_enc).abs()
        equi_loss = equi_diff[src_zeros_f1 == 0].sum()
        equi_diff_map = equi_diff.clone()
        equi_diff_map[src_zeros_f1] = 0

        src_f2_enc = self.model(src_f2)
        tgt_f1_enc = self.model(tgt_f1)

        pre_diff = (src_f1_enc - tgt_f1_enc).abs()

        pre_tissue_mask = (
            tensor_ops.mask.kornia_dilation(tgt_zeros_f1 + src_zeros_f1, width=5) == 0
        )
        pre_loss = pre_diff[..., pre_tissue_mask].sum()
        pre_diff_masked = pre_diff.clone()
        pre_diff_masked[..., pre_tissue_mask == 0] = 0

        post_tissue_mask = (
            tensor_ops.mask.kornia_dilation(tgt_zeros_f1 + src_zeros_f2, width=5) == 0
        )

        post_magn_mask = seed_field.abs().max(1)[0] > self.field_magn_thr
        post_magn_mask[..., 0:10, :] = 0
        post_magn_mask[..., -10:, :] = 0
        post_magn_mask[..., :, 0:10] = 0
        post_magn_mask[..., :, -10:] = 0

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
            log_results(
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
                field=torch.tensor(seed_field),
                equi_diff_map=equi_diff_map,
                post_diff_masked=post_diff_masked,
                pre_diff_masked=pre_diff_masked,
            )
        return loss

    def compute_metroem_loss_old(
        self, batch: dict, mode: str, log_row: bool, sample_name: str = ""
    ):
        src = batch["images"]["src"]
        tgt = batch["images"]["tgt"]

        field = batch["field"]

        tgt_zeros = tensor_ops.mask.kornia_dilation(tgt == self.zero_value, width=3)
        src_zeros = tensor_ops.mask.kornia_dilation(src == self.zero_value, width=3)

        pre_tissue_mask = (src_zeros + tgt_zeros) == 0
        if pre_tissue_mask.sum() / src.numel() < 0.4:
            return None

        zero_magns = 0
        tgt_enc = self.model(tgt)
        zero_magns += tgt_enc[tgt_zeros].abs().sum()

        src_warped = field.field().from_pixels()(src)
        src_warped_enc = self.model(src_warped)
        src_zeros_warped = field.field().from_pixels()(src_zeros.float()) > 0.1

        zero_magns += src_warped_enc[src_zeros_warped].abs().sum()

        # src_enc = (~(field.field().from_pixels()))(src_warped_enc)
        src_enc = self.model(src)

        pre_diff = (src_enc - tgt_enc).abs()
        pre_loss = pre_diff[..., pre_tissue_mask].sum()
        pre_diff_masked = pre_diff.clone()
        pre_diff_masked[..., pre_tissue_mask == 0] = 0

        post_tissue_mask = (
            tensor_ops.mask.kornia_dilation(src_zeros_warped + tgt_zeros, width=5) == 0
        )
        post_magn_mask = field.abs().sum(1) > self.field_magn_thr

        post_magn_mask[..., 0:10, :] = 0
        post_magn_mask[..., -10:, :] = 0
        post_magn_mask[..., :, 0:10] = 0
        post_magn_mask[..., :, -10:] = 0
        post_diff_map = (src_warped_enc - tgt_enc).abs()
        post_mask = post_magn_mask * post_tissue_mask
        post_diff_masked = post_diff_map.clone()
        post_diff_masked[..., post_tissue_mask == 0] = 0
        if post_mask.sum() < 256:
            return None

        post_loss = post_diff_map[..., post_mask].sum()
        loss = pre_loss - post_loss * self.post_weight + zero_magns * self.zero_conserve_weight
        self.log(f"loss/{mode}", loss, on_step=True, on_epoch=True)
        self.log(f"loss/{mode}_pre", pre_loss, on_step=True, on_epoch=True)
        self.log(f"loss/{mode}_post", post_loss, on_step=True, on_epoch=True)
        self.log(f"loss/{mode}_zcons", zero_magns, on_step=True, on_epoch=True)
        if log_row:
            log_results(
                mode,
                sample_name,
                src=src,
                src_enc=src_enc,
                src_warped_enc=src_warped_enc,
                tgt=tgt,
                tgt_enc=tgt_enc,
                field=field,
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
