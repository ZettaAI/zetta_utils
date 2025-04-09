# pragma: no cover

import random
from typing import List, Optional, Union

import attrs
import pytorch_lightning as pl
import torch
import torchvision
import wandb
from PIL import Image

from zetta_utils import builder, tensor_ops
from zetta_utils.training.lightning.train import distributed_available


@builder.register("EncodingCoarsenerRegime", versions="==0.0.0")
@attrs.mutable(eq=False)
class EncodingCoarsenerRegime(pl.LightningModule):  # pylint: disable=too-many-ancestors
    encoder: torch.nn.Module
    decoder: torch.nn.Module
    lr: float
    apply_counts: List[int] = [1]
    invar_angle_range: List[Union[int, float]] = [1, 180]
    invar_mse_weight: float = 0.0
    diffkeep_angle_range: List[Union[int, float]] = [1, 180]
    diffkeep_weight: float = 0.0
    min_nonz_frac: float = 0.2
    worst_val_loss: float = attrs.field(init=False, default=0)
    worst_val_sample: dict = attrs.field(init=False, factory=dict)
    worst_val_sample_idx: Optional[int] = attrs.field(init=False, default=None)

    def __attrs_pre_init__(self):
        super().__init__()

    def log_results(self, mode: str, title_suffix: str = "", **kwargs):
        if not self.logger:
            return
        self.logger.log_image(
            f"results/{mode}_{title_suffix}_slider",
            images=[wandb.Image(v.squeeze(), caption=k) for k, v in kwargs.items()],
        )
        # images = torchvision.utils.make_grid([img[0] for img, _ in img_spec])
        # caption = ",".join(cap for _, cap in img_spec) + title_suffix
        # wandb.log({f"results/{mode}_row": [wandb.Image(images, caption)]})

    def on_validation_epoch_end(self):
        self.log_results(
            "val",
            "worst",
            **self.worst_val_sample,
        )
        self.worst_val_loss = 0
        self.worst_val_sample = {}
        self.worst_val_sample_idx = None

    def validation_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
        interval = 25
        log_row = batch_idx % interval == 0
        sample_name = f"{batch_idx // interval}"
        data_in = batch["data_in"]

        losses = [
            self.compute_loss(data_in, count, "val", log_row, sample_name=sample_name)
            for count in self.apply_counts
        ]
        losses_clean = [l for l in losses if l is not None]
        loss = sum(losses_clean)
        self.log(
            "loss/train",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=distributed_available(),
            rank_zero_only=True,
        )
        return loss

    def training_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
        data_in = batch["data_in"]
        log_row = batch_idx % 100 == 0

        losses = [
            self.compute_loss(data_in, count, "train", log_row) for count in self.apply_counts
        ]
        losses_clean = [l for l in losses if l is not None]
        loss = sum(losses_clean)
        self.log(
            "loss/train",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=distributed_available(),
            rank_zero_only=True,
        )
        return loss

    def compute_loss(
        self,
        data_in: torch.Tensor,
        apply_count: int,
        mode: str,
        log_row: bool,
        sample_name: str = "",
    ):
        setting_name = f"{mode}_apply{apply_count}"

        enc = data_in
        for _ in range(apply_count):
            enc = self.encoder(enc)

        recons = enc
        for _ in range(apply_count):
            recons = self.decoder(recons)

        loss_map_recons = (data_in - recons) ** 2

        loss_recons = loss_map_recons.mean()
        if log_row:
            self.log_results(
                f"{setting_name}_recons",
                sample_name,
                data_in=data_in,
                naive=tensor_ops.interpolate(
                    data_in,
                    size=(enc.shape[-2], enc.shape[-1]),
                    mode="img",
                    unsqueeze_input_to=4,
                ),
                enc=enc,
                recons=recons,
                loss_map_recons=loss_map_recons,
            )

        self.log(
            f"loss/{setting_name}_recons",
            loss_recons,
            on_step=True,
            on_epoch=True,
            sync_dist=distributed_available(),
            rank_zero_only=True,
        )

        if self.invar_mse_weight > 0:
            loss_inv = self.compute_invar_loss(
                data_in, enc, apply_count, log_row, setting_name, sample_name
            )
            self.log(
                f"loss/{setting_name}_inv",
                loss_inv,
                on_step=True,
                on_epoch=True,
                sync_dist=distributed_available(),
                rank_zero_only=True,
            )
        else:
            loss_inv = 0

        if self.diffkeep_weight > 0:
            loss_diffkeep = self.compute_diffkeep_loss(data_in, enc, log_row, sample_name)
            self.log(
                f"loss/{setting_name}_diffkeep",
                loss_diffkeep,
                on_step=True,
                on_epoch=True,
                sync_dist=distributed_available(),
                rank_zero_only=True,
            )
        else:
            loss_diffkeep = 0

        loss = (
            loss_recons + self.invar_mse_weight * loss_inv + self.diffkeep_weight * loss_diffkeep
        )

        self.log(
            f"loss/{setting_name}",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=distributed_available(),
            rank_zero_only=True,
        )

        if mode == "val":
            if loss > self.worst_val_loss:
                self.worst_val_loss = loss
                self.worst_val_sample = {
                    "data_in": data_in,
                    "enc": enc,
                    "recons": recons,
                    "loss_map": loss_map_recons,
                }

        return loss

    def compute_invar_loss(
        self,
        data_in: torch.Tensor,
        enc: torch.Tensor,
        apply_count: int,
        log_row: bool,
        setting_name: str,
        sample_name: str = "",
    ):
        angle = random.uniform(self.invar_angle_range[0], self.invar_angle_range[1])
        data_in_rot = torchvision.transforms.functional.rotate(
            img=data_in,
            angle=angle,
            interpolation=Image.BILINEAR,
        )
        enc_rot = torchvision.transforms.functional.rotate(
            img=enc,
            angle=angle,
            interpolation=Image.BILINEAR,
        )
        rot_input_enc = data_in_rot
        for _ in range(apply_count):
            rot_input_enc = self.encoder(rot_input_enc)
        loss_map_invar = (enc_rot - rot_input_enc) ** 2
        result = loss_map_invar.mean()
        if log_row:
            self.log_results(
                f"{setting_name}_inv",
                title_suffix=sample_name,
                data_in=data_in,
                rot_input_enc=rot_input_enc,
                enc_rot=enc_rot,
                loss_map_invar=loss_map_invar,
            )
        return result

    def compute_diffkeep_loss(
        self,
        data_in: torch.Tensor,
        enc: torch.Tensor,
        log_row: bool,
        setting_name: str,
        sample_name: str = "",
    ):
        angle = random.uniform(self.diffkeep_angle_range[0], self.diffkeep_angle_range[1])
        data_in_rot = torchvision.transforms.functional.rotate(
            img=data_in,
            angle=angle,
            interpolation=Image.BILINEAR,
        )
        enc_rot = torchvision.transforms.functional.rotate(
            img=enc,
            angle=angle,
            interpolation=Image.BILINEAR,
        )
        data_in_diff = (data_in - data_in_rot) ** 2
        enc_diff = (enc - enc_rot) ** 2
        data_in_diff_downs = tensor_ops.interpolate(
            data_in_diff, size=enc_diff.shape[-2:], mode="img", unsqueeze_input_to=4
        )
        loss_map_diffkeep = (data_in_diff_downs - enc_diff).abs()

        result = loss_map_diffkeep.mean()
        if log_row:
            self.log_results(
                f"{setting_name}_diffkeep",
                title_suffix=sample_name,
                data_in=data_in,
                enc_diff=enc_diff,
                data_in_diff_downs=data_in_diff_downs,
                loss_map_diffkeep=loss_map_diffkeep,
            )
        return result

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
