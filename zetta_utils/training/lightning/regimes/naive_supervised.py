from typing import Optional

import attrs
import pytorch_lightning as pl
import torch

import wandb
from zetta_utils import builder


@builder.register("NaiveSupervised")
@attrs.mutable(eq=False)
class NaiveSupervised(pl.LightningModule):  # pylint: disable=too-many-ancestors
    model: torch.nn.Module
    lr: float
    min_nonz_frac: float = 0.2
    worst_val_loss: float = attrs.field(init=False, default=0)
    worst_val_sample: dict = attrs.field(init=False, factory=dict)
    worst_val_sample_idx: Optional[int] = attrs.field(init=False, default=None)

    def __attrs_pre_init__(self):
        super().__init__()

    @staticmethod
    def log_results(
        mode: str,
        data_in: torch.Tensor,
        target: torch.Tensor,
        result: torch.Tensor,
        loss_map: torch.Tensor,
        title_suffix: str = "",
    ):
        img_spec = [
            (data_in, "Data In"),
            (target, "Target"),
            (result, "Result"),
            (loss_map, "Loss Map"),
        ]

        wandb.log(
            {
                f"results/{mode}_slider": [
                    wandb.Image(img.squeeze(), caption=cap + title_suffix) for img, cap in img_spec
                ]
            }
        )
        # images = torchvision.utils.make_grid([img[0] for img, _ in img_spec])
        # caption = ",".join(cap for _, cap in img_spec) + title_suffix
        # wandb.log({f"results/{mode}_row": [wandb.Image(images, caption)]})

    def validation_epoch_end(self, _):
        self.log_results(
            "val_worst",
            title_suffix=f" (idx={self.worst_val_sample_idx})",
            **self.worst_val_sample,
        )
        self.worst_val_loss = 0
        self.worst_val_sample = {}
        self.worst_val_sample_idx = None

    def validation_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
        data_in = batch["data_in"]
        target = batch["target"]
        if (data_in != 0).sum() / data_in.numel() < self.min_nonz_frac:
            loss = None
        else:
            result = self.model(data_in)
            loss_map = (target - result) ** 2
            if "loss_weights" in batch:
                loss_weights = batch["loss_weights"]
                loss = (loss_map * loss_weights).mean()
            else:
                loss = loss_map.mean()
            self.log("loss/val", loss, on_step=True, on_epoch=True)
            if loss > self.worst_val_loss:
                self.worst_val_loss = loss
                self.worst_val_sample = {
                    "data_in": data_in,
                    "result": result,
                    "target": target,
                    "loss_map": loss_map,
                }
                self.worst_val_sample_idx = batch_idx

            interval = 25
            if batch_idx % interval == 0:
                self.log_results(
                    f"val_{batch_idx // interval}",
                    data_in=data_in,
                    target=target,
                    result=result,
                    loss_map=loss_map,
                )
        return loss

    def training_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
        data_in = batch["data_in"]
        target = batch["target"]

        if (data_in != 0).sum() == 0:
            loss = None
        else:
            result = self.model(data_in)
            loss_map = (target - result) ** 2
            if "loss_weights" in batch:
                loss_weights = batch["loss_weights"]
                loss = (loss_map * loss_weights).mean()
            else:
                loss = loss_map.mean()

            self.log("loss/train", loss, on_step=True, on_epoch=True)
            if batch_idx % 500 == 0:
                self.log_results(
                    "train", data_in=data_in, target=target, result=result, loss_map=loss_map
                )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
