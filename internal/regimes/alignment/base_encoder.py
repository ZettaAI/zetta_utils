# pragma: no cover
# pylint: disable=too-many-locals

import os
from math import log2
from typing import Optional

import attrs
import einops
import numpy as np
import pytorch_lightning as pl
import torch
import torchfields
import wandb
from PIL import Image as PILImage
from pytorch_lightning import seed_everything

from zetta_utils import builder, distributions, tensor_ops, viz


@builder.register("BaseEncoderRegime", versions="==0.0.2")
@attrs.mutable(eq=False)
class BaseEncoderRegime(pl.LightningModule):  # pylint: disable=too-many-ancestors
    model: torch.nn.Module
    lr: float
    train_log_row_interval: int = 200
    val_log_row_interval: int = 25
    max_displacement_px: float = 16.0
    l1_weight_start_val: float = 0.0
    l1_weight_end_val: float = 0.0
    l1_weight_start_epoch: int = 0
    l1_weight_end_epoch: int = 0
    locality_weight: float = 1.0
    similarity_weight: float = 0.0
    zero_value: float = 0
    ds_factor: int = 1
    worst_val_loss: float = attrs.field(init=False, default=0)
    worst_val_sample: dict = attrs.field(init=False, factory=dict)
    worst_val_sample_idx: Optional[int] = attrs.field(init=False, default=None)

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
        if not self.logger:
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
                    if v.size(1) == 2 and k != "field":
                        img = torch.cat([v, torch.zeros_like(v[:, :1])], dim=1)
                    else:
                        img = v
                    v_min = img[b].min().round(decimals=4)
                    v_max = img[b].max().round(decimals=4)
                    images.append(
                        wandb.Image(
                            viz.rendering.Renderer()(img[b].squeeze()),
                            caption=f"{k}_b{b} | min: {v_min} | max: {v_max}",
                        )
                    )

        self.logger.log_image(f"results/{mode}_{title_suffix}_slider", images=images)

    def validation_epoch_start(self, _):  # pylint: disable=no-self-use
        seed_everything(42)

    def on_validation_epoch_end(self):
        env_seed = os.environ.get("PL_GLOBAL_SEED")
        if env_seed is not None:
            seed_everything(int(env_seed) + self.current_epoch)
        else:
            seed_everything(None)

    def training_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
        log_row = batch_idx % self.train_log_row_interval == 0

        with torchfields.set_identity_mapping_cache(True, clear_cache=False):
            loss = self.compute_metroem_loss(batch=batch, mode="train", log_row=log_row)

        return loss

    @staticmethod
    def _get_warped(img, field=None):
        if field is not None:
            img_warped = field.from_pixels()(img)
        else:
            img_warped = img

        return img_warped

    @staticmethod
    def _down_zeros_mask(zeros_mask, count):
        if count <= 0:
            return zeros_mask

        scale_factor = 0.5 ** count
        return (
            torch.nn.functional.interpolate(
                zeros_mask.float(), scale_factor=scale_factor, mode="bilinear"
            )
            > 0.99
        )

    def compute_metroem_loss(self, batch: dict, mode: str, log_row: bool, sample_name: str = ""):
        src = batch["images"]["src_img"]
        tgt = batch["images"]["tgt_img"]

        # if (
        #     (src == self.zero_value) + (tgt == self.zero_value)
        # ).bool().sum() / src.numel() > self.empty_tissue_threshold:
        #     return None  # Can't return None with DDP!

        # Get random field - combination of pregenerated Perlin noise and a random affine transform
        seed_field = batch["field"].field_()
        f_warp = seed_field * self.max_displacement_px
        f_aff = (
            einops.rearrange(
                tensor_ops.generators.get_affine_field(
                    size=src.shape[-1],
                    rot_deg=self.equivar_rot_deg_distr() if mode == "train" else 90.0,
                    scale=self.equivar_scale_distr() if mode == "train" else 1.0,
                    shear_x_deg=self.equivar_shear_deg_distr() if mode == "train" else 0.0,
                    shear_y_deg=self.equivar_shear_deg_distr() if mode == "train" else 0.0,
                    trans_x_px=self.equivar_trans_px_distr() if mode == "train" else 0.0,
                    trans_y_px=self.equivar_trans_px_distr() if mode == "train" else 0.0,
                ),
                "C X Y Z -> Z C X Y",
            )
            .pixels()  # type: ignore
            .to(seed_field.device)
        ).repeat_interleave(src.size(0), dim=0)
        f1_transform = f_aff.from_pixels()(f_warp.from_pixels()).pixels()

        # Warp Images and Tissue mask
        src_f1 = self._get_warped(src, field=f1_transform)
        tgt_f1 = self._get_warped(tgt, field=f1_transform)

        # Generate encodings: src, src_f1_enc, src_enc_f1, tgt_f1_enc
        src_enc = self.model(src)
        src_enc_f1 = torch.nn.functional.pad(src_enc, (1, 1, 1, 1), mode="replicate")
        src_enc_f1 = (
            torch.nn.functional.pad(f1_transform, (self.ds_factor,) * 4, mode="replicate")
            .from_pixels()  # type: ignore[attr-defined]
            .down(int(log2(self.ds_factor)))
            .sample(src_enc_f1, padding_mode="border")
        )
        src_enc_f1 = torch.nn.functional.pad(src_enc_f1, (-1, -1, -1, -1))
        tgt_f1_enc = self.model(tgt_f1)

        crop = 256 // self.ds_factor
        src_f1 = src_f1[..., 256:-256, 256:-256]
        tgt_f1 = tgt_f1[..., 256:-256, 256:-256]
        src_enc_f1 = src_enc_f1[..., crop:-crop, crop:-crop]
        tgt_f1_enc = tgt_f1_enc[..., crop:-crop, crop:-crop]

        # Alignment loss: Ensure even close to local optima solutions produce larger errors
        # than the local optimum solution
        abs_error_local_opt = (
            (src_enc_f1 - tgt_f1_enc)[:, :, 1:-1, 1:-1].pow(2).mean(dim=-3, keepdim=True)
        )
        abs_error_1px_shift = torch.stack(
            [
                (src_enc_f1[:, :, 2:, 1:-1] - tgt_f1_enc[:, :, 1:-1, 1:-1]).pow(2),
                (src_enc_f1[:, :, :-2, 1:-1] - tgt_f1_enc[:, :, 1:-1, 1:-1]).pow(2),
                (src_enc_f1[:, :, 1:-1, 2:] - tgt_f1_enc[:, :, 1:-1, 1:-1]).pow(2),
                (src_enc_f1[:, :, 1:-1, :-2] - tgt_f1_enc[:, :, 1:-1, 1:-1]).pow(2),
                (tgt_f1_enc[:, :, 2:, 2:] - src_enc_f1[:, :, 1:-1, 1:-1]).pow(2),
                (tgt_f1_enc[:, :, :-2, 2:] - src_enc_f1[:, :, 1:-1, 1:-1]).pow(2),
                (tgt_f1_enc[:, :, 2:, :-2] - src_enc_f1[:, :, 1:-1, 1:-1]).pow(2),
                (tgt_f1_enc[:, :, :-2, :-2] - src_enc_f1[:, :, 1:-1, 1:-1]).pow(2),
            ]
        ).mean(dim=-3, keepdim=True)

        locality_error_map = (
            ((abs_error_local_opt - abs_error_1px_shift + 4.0) * 0.2)
            .pow(
                8.0  # increase to put more focus on locations where bad alignment
                # still produces similar encodings - try 8? -> 42
            )
            .logsumexp(dim=0)
        )

        locality_loss = (
            locality_error_map.sum() / locality_error_map.size(0) * self.ds_factor * self.ds_factor
        )

        l1_loss_map = (tgt_f1_enc.abs() + src_enc_f1.abs())[:, :, 1:-1, 1:-1]
        l1_loss = (
            l1_loss_map.sum()
            / (2 * tgt_f1_enc.size(0) * tgt_f1_enc.size(1))
            * self.ds_factor
            * self.ds_factor
        )

        l1_weight_ratio = min(
            1.0,
            max(0, self.current_epoch - self.l1_weight_start_epoch)
            / max(1, self.l1_weight_end_epoch - self.l1_weight_start_epoch),
        )
        l1_weight = (
            l1_weight_ratio * self.l1_weight_end_val
            + (1.0 - l1_weight_ratio) * self.l1_weight_start_val
        )

        loss = locality_loss * self.locality_weight + l1_loss * l1_weight
        self.log(
            f"loss/{mode}", loss, on_step=True, on_epoch=True, sync_dist=True, rank_zero_only=False
        )
        self.log(
            f"loss/{mode}_l1_weight",
            l1_weight,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=False,
            rank_zero_only=True,
        )
        self.log(
            f"loss/{mode}_l1",
            l1_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            rank_zero_only=False,
        )
        if log_row:
            self.log_results(
                mode,
                sample_name,
                src=src,
                src_enc=src_enc,
                src_f1=src_f1,
                src_enc_f1=src_enc_f1,
                tgt_f1=tgt_f1,
                tgt_f1_enc=tgt_f1_enc,
                field=f_warp.tensor_(),
                locality_error_map=locality_error_map,
                l1_loss_map=l1_loss_map,
                weighted_loss_map=(
                    locality_error_map / locality_error_map.size(0) * self.locality_weight
                    + l1_loss_map / (2 * tgt_f1_enc.size(0)) * l1_weight
                ),
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
