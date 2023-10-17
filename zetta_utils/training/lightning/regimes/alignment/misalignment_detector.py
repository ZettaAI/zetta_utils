# pragma: no cover

import random
from typing import Optional

import attrs
import pytorch_lightning as pl
import torch
import torchvision
import wandb

import zetta_utils as zu
from zetta_utils import builder, convnet, tensor_ops  # pylint: disable=unused-import


@builder.register("MisalignmentDetectorRegime")
@attrs.mutable(eq=False)
class MisalignmentDetectorRegime(pl.LightningModule):  # pylint: disable=too-many-ancestors
    detector: torch.nn.Module
    lr: float
    max_disp: float
    downsample_power: int = 0
    min_nonz_frac: float = 0.2
    worst_val_loss: float = attrs.field(init=False, default=0)
    worst_val_sample: dict = attrs.field(init=False, default=attrs.Factory(dict))
    worst_val_sample_idx: Optional[int] = attrs.field(init=False, default=None)

    def __attrs_pre_init__(self):
        super().__init__()

    # TODO: factor this out
    @staticmethod
    def augment_field(field):
        random_vertical_flip = torchvision.transforms.RandomVerticalFlip()
        random_horizontal_flip = torchvision.transforms.RandomHorizontalFlip()
        angle = random.choice([0, 90, 180, 270])
        return random_horizontal_flip(
            random_vertical_flip(torchvision.transforms.functional.rotate(field, angle))
        )

    @staticmethod
    def norm_field(field, threshold):
        field_i = field[:, 0:1, :, :]
        field_j = field[:, 1:2, :, :]
        field_norm = (field_i ** 2 + field_j ** 2) ** 0.5
        return torch.clamp(field_norm, 0, threshold)

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

        image = batch["image"]["data_in"]
        # field0 = torchfields.Field(batch["field0"]["data_in"])
        # field1 = torchfields.Field(batch["field1"]["data_in"])
        field0 = batch["field0"]["data_in"]
        field1 = batch["field1"]["data_in"]

        loss = self.compute_loss(image, field0, field1, "validate", log_row)
        if loss is not None:
            self.log("loss/train", loss, on_step=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
        image = batch["image"]["data_in"]
        field0 = batch["field0"]["data_in"]
        field1 = batch["field1"]["data_in"]
        log_row = batch_idx % 100 == 0

        loss = self.compute_loss(image, field0, field1, "train", log_row)
        if loss is not None:
            self.log("loss/train", loss, on_step=True, on_epoch=True)
        return loss

    def compute_loss(
        self,
        image: torch.Tensor,
        field0: torch.Tensor,
        field1: torch.Tensor,
        mode: str,
        log_row: bool,
        sample_name: str = "",
    ):
        setting_name = f"{mode}"

        if (image != 0).sum() / image.numel() < self.min_nonz_frac:
            loss = None
        else:
            with torch.no_grad():
                field0 = self.augment_field(field0)
                field1 = self.augment_field(field1)
                labels = self.norm_field(field0 - field1, self.max_disp)
                if self.downsample_power != 0:
                    image = zu.tensor_ops.interpolate(
                        image,
                        size=(
                            image.shape[-2] // 2 ** self.downsample_power,
                            image.shape[-1] // 2 ** self.downsample_power,
                        ),
                        mode="img",
                    )
                    field0 = zu.tensor_ops.interpolate(
                        field0,
                        size=(
                            field0.shape[-2] // 2 ** self.downsample_power,
                            field0.shape[-1] // 2 ** self.downsample_power,
                        ),
                        mode="img",
                    )
                    field1 = zu.tensor_ops.interpolate(
                        field1,
                        size=(
                            field1.shape[-2] // 2 ** self.downsample_power,
                            field1.shape[-1] // 2 ** self.downsample_power,
                        ),
                        mode="img",
                    )
                    labels = zu.tensor_ops.interpolate(
                        labels,
                        size=(
                            labels.shape[-2] // 2 ** self.downsample_power,
                            labels.shape[-1] // 2 ** self.downsample_power,
                        ),
                        mode="img",
                    )
                # fields are typed as Tensor
                assert hasattr(field0, "field_")
                assert hasattr(field1, "field_")
                image[:, 0:1, :, :] = field0.field_().from_pixels()(image[:, 0:1, :, :])
                image[:, 1:2, :, :] = field1.field_().from_pixels()(image[:, 1:2, :, :])
            pred_labels = self.detector(image)

            loss_map = (pred_labels - labels) ** 2
            loss = loss_map.mean()

            if log_row:
                self.log_results(
                    f"{setting_name}_recons",
                    sample_name,
                    image0=image[0, 0, :, :].detach().cpu(),
                    image1=image[0, 1, :, :].detach().cpu(),
                    #                    field0=field0,
                    #                    field1=field1,
                    pred_labels=pred_labels[0, :, :, :].detach().cpu(),
                    labels=labels[0, :, :, :].detach().cpu(),
                    loss_map=loss_map[0, :, :, :].detach().cpu(),
                )

            self.log(f"loss/{setting_name}", loss, on_step=True, on_epoch=True)

            if mode == "val":
                if loss > self.worst_val_loss:
                    self.worst_val_loss = loss
                    self.worst_val_sample = {
                        "image": image,
                        "pred_labels": pred_labels,
                        "labels": labels,
                        "loss_map": loss_map,
                    }

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
