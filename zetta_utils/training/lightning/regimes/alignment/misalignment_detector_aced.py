# pylint: disable=too-many-locals
import os
from typing import Literal, Optional

import attrs
import cc3d
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from PIL import Image as PILImage
from pytorch_lightning import seed_everything

from zetta_utils import builder, convnet, distributions, viz


@builder.register("MisalignmentDetectorAcedRegime", allow_parallel=False)
@attrs.mutable(eq=False)
class MisalignmentDetectorAcedRegime(pl.LightningModule):  # pylint: disable=too-many-ancestors
    model: torch.nn.Module
    lr: float
    train_log_row_interval: int = 200
    val_log_row_interval: int = 25
    field_magn_thr: float = 1

    max_shared_displacement_px: float = 8.0
    max_src_displacement_px: distributions.Distribution = distributions.uniform_distr(8.0, 32.0)
    equivar_rot_deg_distr: distributions.Distribution = distributions.uniform_distr(0, 360)
    equivar_trans_px_distr: distributions.Distribution = distributions.uniform_distr(-10, 10)

    tgt_val_translation: int = 4
    zero_value: float = 0
    output_mode: Literal["binary", "displacement"] = "binary"

    encoder_path: Optional[str] = None
    encoder: torch.nn.Module = attrs.field(init=False, default=torch.nn.Identity())

    worst_val_loss: float = attrs.field(init=False, default=0)
    worst_val_sample: dict = attrs.field(init=False, factory=dict)
    worst_val_sample_idx: Optional[int] = attrs.field(init=False, default=None)

    def __attrs_pre_init__(self):
        super().__init__()

    def __attrs_post_init__(self):
        if self.encoder_path is not None:
            self.encoder = convnet.utils.load_model(self.encoder_path, use_cache=True).eval()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
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
        env_seed = os.environ.get("PL_GLOBAL_SEED")
        if env_seed is not None:
            seed_everything(int(env_seed) + self.current_epoch)
        else:
            seed_everything(None)

    def _get_warped(self, img, field=None):
        img_padded = torch.nn.functional.pad(img, (1, 1, 1, 1), value=self.zero_value)
        if field is not None:
            assert hasattr(field, "from_pixels")  # mypy torchfields compatibility
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
                .from_pixels()  # type: ignore
                .sample((~zeros_padded).float(), padding_mode="border")
                <= 0.1
            )
            non_tissue_zeros_warped = (
                torch.nn.functional.pad(field, (1, 1, 1, 1), mode="replicate")
                .from_pixels()  # type: ignore
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

    def training_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
        log_row = batch_idx % self.train_log_row_interval == 0
        losses = [
            self.compute_misd_loss(
                batch=batch,
                mode="train",
                log_row=log_row,
            )
        ]
        losses_clean = [l for l in losses if l is not None]
        if len(losses_clean) == 0:
            return None
        loss = sum(losses_clean)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
        with torch.no_grad():
            log_row = batch_idx % self.val_log_row_interval == 0
            sample_name = f"{batch_idx // self.val_log_row_interval}"

            losses = [
                self.compute_misd_loss(
                    batch=batch, mode="val", log_row=log_row, sample_name=sample_name
                )
            ]
            losses_clean = [l for l in losses if l is not None]
            if len(losses_clean) == 0:
                return None
            loss = sum(losses_clean)
            return loss

    def compute_misd_loss(self, batch: dict, mode: str, log_row: bool, sample_name: str = ""):
        src = batch["images"]["src"]
        tgt = batch["images"]["tgt"]

        if ((src == self.zero_value) + (tgt == self.zero_value)).bool().sum() / src.numel() > 0.7:
            return None

        gt_displacement = batch["images"]["displacement"]
        gt_labels = gt_displacement.clone()

        if self.output_mode == "binary":
            gt_labels = gt_labels > self.field_magn_thr

        src_warped, src_warped_tissue_wo_soma, src_warped_tissue_w_soma = self._get_warped(
            src, field=None
        )
        tgt_warped, tgt_warped_tissue_wo_soma, tgt_warped_tissue_w_soma = self._get_warped(
            tgt, field=None
        )

        # Create mask that excludes soma interior from loss, but keep thin tissue in between
        # from either section
        joint_tissue = tgt_warped_tissue_wo_soma + src_warped_tissue_wo_soma
        # Previous mask also added partial tissue at boundary - don't want to penalize there
        intersect_tissue = joint_tissue & src_warped_tissue_w_soma & tgt_warped_tissue_w_soma
        if intersect_tissue.sum() == 0:
            return None

        with torch.no_grad():
            src_encoded = self.encoder(src_warped)
            tgt_encoded = self.encoder(tgt_warped)
        prediction = self.model(torch.cat((src_encoded, tgt_encoded), 1))

        fg_ratio = (gt_labels & intersect_tissue).sum() / intersect_tissue.sum()
        if 0.0 < fg_ratio < 1.0:
            weight = (1.0 - fg_ratio) * gt_labels + fg_ratio * ~gt_labels
        else:
            weight = torch.ones_like(gt_labels, dtype=torch.float32)
        weight[intersect_tissue == 0] = 0.0

        loss_map = torch.nn.functional.binary_cross_entropy_with_logits(
            prediction, gt_labels.float(), weight=weight, reduction="none"
        )

        loss = loss_map[intersect_tissue].sum() / loss_map.size(0)

        self.log(f"loss/{mode}", loss.item(), on_step=True, on_epoch=True)

        if log_row:
            self.log_results(
                mode,
                sample_name,
                src=src,
                tgt=tgt,
                final_tissue=intersect_tissue,
                gt_displacement=gt_displacement,
                gt_labels=gt_labels,
                weight=weight,
                prediction=prediction,
                loss_map=loss_map,
            )
        return loss
