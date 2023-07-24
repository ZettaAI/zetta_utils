# pragma: no cover

import random
from typing import List, Optional

import attrs
import PIL
import pytorch_lightning as pl
import torch
import torchfields
import torchvision
import wandb

import zetta_utils as zu
from zetta_utils import builder, convnet, tensor_ops  # pylint: disable=unused-import


# TODO: Refactor function
def warp_by_px(image, direction, pixels):

    fields = torch.zeros(
        1, 2, image.shape[-2], image.shape[-1], device=image.device
    ).field()  # type: ignore

    if direction == 0:
        fields[0, 0, :, :] = 0
        fields[0, 1, :, :] = pixels
    elif direction == 1:
        fields[0, 0, :, :] = pixels
        fields[0, 1, :, :] = 0
    elif direction == 2:
        fields[0, 0, :, :] = 0
        fields[0, 1, :, :] = -pixels
    elif direction == 3:
        fields[0, 0, :, :] = -pixels
        fields[0, 1, :, :] = 0
    elif direction == 4:
        fields[0, 0, :, :] = pixels ** 0.5
        fields[0, 1, :, :] = pixels ** 0.5
    elif direction == 5:
        fields[0, 0, :, :] = -(pixels ** 0.5)
        fields[0, 1, :, :] = pixels ** 0.5
    elif direction == 6:
        fields[0, 0, :, :] = pixels ** 0.5
        fields[0, 1, :, :] = -(pixels ** 0.5)
    elif direction == 7:
        fields[0, 0, :, :] = -(pixels ** 0.5)
        fields[0, 1, :, :] = -(pixels ** 0.5)
    return (
        fields.from_pixels().expand(
            image.shape[0], fields.shape[1], fields.shape[2], fields.shape[3]
        )
    )(image)


# TODO: Refactor function
def center_crop_norm(image):
    norm = torchvision.transforms.Normalize(0, 1)
    crop = torchvision.transforms.CenterCrop(image.shape[-2] // 2)
    return crop(norm(image))


@builder.register("EncodingCoarsenerHighRes")
@attrs.mutable(eq=False)
class EncodingCoarsenerHighRes(pl.LightningModule):  # pylint: disable=too-many-ancestors
    encoder: torch.nn.Module
    decoder: torch.nn.Module
    lr: float
    encoder_ckpt_path: Optional[str] = None
    decoder_ckpt_path: Optional[str] = None
    apply_counts: List[int] = [1]
    residual_range: List[float] = [0.1, 5.0]
    residual_weight: float = 0.0
    field_scale: List[float] = [1.0, 1.0]
    field_weight: float = 0.0
    meanstd_weight: float = 0.0
    invar_weight: float = 0.0
    min_nonz_frac: float = 0.2
    worst_val_loss: float = attrs.field(init=False, default=0)
    worst_val_sample: dict = attrs.field(init=False, default=attrs.Factory(dict))
    worst_val_sample_idx: Optional[int] = attrs.field(init=False, default=None)

    def __attrs_pre_init__(self):
        super().__init__()

    def __attrs_post_init__(self):
        if self.encoder_ckpt_path is not None:
            convnet.utils.load_weights_file(self, self.encoder_ckpt_path, ["encoder"])

        if self.decoder_ckpt_path is not None:
            convnet.utils.load_weights_file(self, self.decoder_ckpt_path, ["decoder"])

    @staticmethod
    def log_results(mode: str, title_suffix: str = "", **kwargs):
        wandb.log(
            {
                f"results/{mode}_{title_suffix}_slider": [
                    wandb.Image(v.squeeze(), caption=k) for k, v in kwargs.items()
                ]
            }
        )
        # images = torchvision.utils.make_grid([img[0] for img, _ in img_spec])
        # caption = ",".join(cap for _, cap in img_spec) + title_suffix
        # wandb.log({f"results/{mode}_row": [wandb.Image(images, caption)]})

    def validation_epoch_end(self, _):
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
        data_in = batch["image"]["data_in"]
        field_in = batch["field"]["data_in"]

        losses = [
            self.compute_loss(data_in, field_in, count, "val", log_row, sample_name=sample_name)
            for count in self.apply_counts
        ]
        losses_clean = [l for l in losses if l is not None]
        if len(losses_clean) == 0:
            loss = None
        else:
            loss = sum(losses_clean)
            self.log("loss/train", loss, on_step=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
        data_in = batch["image"]["data_in"]
        field_in = batch["field"]["data_in"]
        log_row = batch_idx % 100 == 0

        losses = [
            self.compute_loss(data_in, field_in, count, "train", log_row)
            for count in self.apply_counts
        ]
        losses_clean = [l for l in losses if l is not None]
        if len(losses_clean) == 0:
            loss = None
        else:
            loss = sum(losses_clean)
            self.log("loss/train", loss, on_step=True, on_epoch=True)
        return loss

    def compute_loss(  # pylint: disable=too-many-locals, too-many-branches
        self,
        data_in: torch.Tensor,
        field_in: torchfields.Field,
        apply_count: int,
        mode: str,
        log_row: bool,
        sample_name: str = "",
    ):
        setting_name = f"{mode}_apply{apply_count}"

        if (data_in != 0).sum() / data_in.numel() < self.min_nonz_frac:
            loss = None
        else:
            enc = data_in
            for _ in range(apply_count):
                enc = self.encoder(enc)

            recons = enc
            for _ in range(apply_count):
                recons = self.decoder(recons)

            mean_loss = (torch.mean(data_in) - torch.mean(enc)) ** 2
            std_loss = (torch.std(data_in) - torch.std(enc)) ** 2

            loss_map_recons = (data_in - recons) ** 2

            loss_recons = loss_map_recons.mean()

            if log_row:
                self.log_results(
                    f"{setting_name}_recons",
                    sample_name,
                    data_in=data_in[0:1, :, :, :],
                    naive=zu.tensor_ops.interpolate(
                        data_in[0:1, :, :, :], size=(enc.shape[-2], enc.shape[-1]), mode="img"
                    ),
                    enc=enc[0:1, :, :, :],
                    recons=recons[0:1, :, :, :],
                    loss_map_recons=loss_map_recons[0:1, :, :, :],
                )

            self.log(f"loss/{setting_name}_recons", loss_recons, on_step=True, on_epoch=True)

            field_in *= random.uniform(self.field_scale[0], self.field_scale[1])

            if self.field_weight > 0:
                loss_field = self.compute_field_loss(
                    data_in, field_in, enc, apply_count, log_row, setting_name, sample_name
                )
                self.log(f"loss/{setting_name}_field", loss_field, on_step=True, on_epoch=True)
            else:
                loss_field = 0

            if self.invar_weight > 0:
                loss_invar = self.compute_invar_loss(
                    data_in, enc, apply_count, log_row, setting_name, sample_name
                )
                self.log(f"loss/{setting_name}_invar", loss_invar, on_step=True, on_epoch=True)
            else:
                loss_invar = 0

            if self.residual_weight > 0:
                loss_res = self.compute_residual_loss(
                    data_in, enc, apply_count, log_row, setting_name, sample_name
                )
                self.log(f"loss/{setting_name}_res", loss_res, on_step=True, on_epoch=True)
            else:
                loss_res = 0

            loss = (
                loss_recons
                + self.meanstd_weight * mean_loss
                + self.meanstd_weight * std_loss
                + +self.residual_weight * loss_res
                + self.field_weight * loss_field
                + self.invar_weight * loss_invar
            )

            self.log(f"loss/{setting_name}", loss, on_step=True, on_epoch=True)

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

    def compute_field_loss(
        self,
        data_in: torch.Tensor,
        field_in: torchfields.Field,
        enc: torch.Tensor,
        apply_count: int,
        log_row: bool,
        setting_name: str,
        sample_name: str = "",
    ):
        field_in_apply = 2 * torchfields.Field(field_in.cpu()).cuda().from_pixels()
        enc_warped = field_in_apply(data_in)
        for _ in range(apply_count):
            enc_warped = self.encoder(enc_warped)
        warped_enc = torch.nn.functional.interpolate(
            field_in_apply, scale_factor=(1 / 2) ** apply_count, mode="bilinear"
        )(enc)

        loss_field = (warped_enc - enc_warped) ** 2

        result = loss_field.mean()
        if log_row:
            self.log_results(
                f"{setting_name}_field",
                title_suffix=sample_name,
                data_in=data_in[0],
                field_in=field_in[0],
                enc=enc[0],
                enc_warped=enc_warped[0],
                warped_enc=warped_enc[0],
                loss_field=loss_field[0],
            )
        return result

    def compute_residual_loss(  # pylint: disable=too-many-locals
        self,
        data_in: torch.Tensor,
        enc: torch.Tensor,
        apply_count: int,
        log_row: bool,
        setting_name: str,
        sample_name: str = "",
    ):
        px_a = random.uniform(self.residual_range[0], self.residual_range[1])
        px_b = random.uniform(self.residual_range[0], self.residual_range[1])

        px_a *= 2 ** apply_count
        px_b *= 2 ** apply_count

        direction = random.choice([0, 1, 2, 3, 4, 5, 6, 7])
        outputs_a = warp_by_px(data_in, direction, px_a)
        outputs_b = warp_by_px(data_in, direction, px_b)
        for _ in range(apply_count):
            outputs_a = self.encoder(outputs_a)
            outputs_b = self.encoder(outputs_b)
        outputs_a = center_crop_norm(outputs_a)
        outputs_b = center_crop_norm(outputs_b)
        encodings = center_crop_norm(enc).expand(outputs_a.shape)

        loss_a = (encodings - outputs_a) ** 2
        loss_b = (encodings - outputs_b) ** 2

        loss_res = (loss_a.mean() * px_a - loss_b.mean() / px_b) ** 2

        result = loss_res.mean()
        if log_row:
            self.log_results(
                f"{setting_name}_res",
                title_suffix=sample_name,
                data_in=data_in,
                loss_a=loss_a,
                loss_b=loss_b,
            )
        return result

    def compute_invar_loss(
        self,
        data_in: torch.Tensor,
        enc: torch.Tensor,
        apply_count: int,
        log_row: bool,
        setting_name: str,
        sample_name: str = "",
    ):
        angle = random.uniform(-180, 180)
        data_in_rot = torchvision.transforms.functional.rotate(
            img=data_in,
            angle=angle,
            interpolation=PIL.Image.BILINEAR,
        )
        enc_rot = torchvision.transforms.functional.rotate(
            img=enc,
            angle=angle,
            interpolation=PIL.Image.BILINEAR,
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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
