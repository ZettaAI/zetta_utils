# pylint: disable=arguments-differ,no-self-use,too-many-ancestors
import attrs
import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything

from zetta_utils import builder

from .common import log_results


@builder.register("NaiveSupervisedRegime")
@attrs.mutable(eq=False)
class NaiveSupervisedRegime(pl.LightningModule):
    model: torch.nn.Module
    lr: float
    min_nonz_frac: float = 0.2

    train_log_row_interval: int = 200
    val_log_row_interval: int = 25

    def __attrs_pre_init__(self):
        super().__init__()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def validation_epoch_start(self, _):
        seed_everything(42)

    def on_validation_epoch_end(self):
        seed_everything(None)

    def validation_step(self, batch, batch_idx):
        log_row = batch_idx % self.val_log_row_interval == 0
        sample_name = f"{batch_idx // self.val_log_row_interval}"
        loss = self.compute_loss(batch=batch, mode="val", log_row=log_row, sample_name=sample_name)
        return loss

    def compute_loss(self, batch: dict, mode: str, log_row: bool, sample_name: str = ""):
        data_in = batch["data_in"]
        target = batch["target"]

        if (data_in != 0).sum() / data_in.numel() < self.min_nonz_frac:
            return None

        result = self.model(data_in)
        loss_map = (target - result) ** 2

        if "loss_weights" in batch:
            loss_weights = batch["loss_weights"]
            loss = (loss_map * loss_weights).mean()
        else:
            loss = loss_map.sum()
        self.log(f"loss/{mode}", loss.item(), on_step=True, on_epoch=True)

        if log_row:
            log_results(
                mode,
                sample_name,
                data_in=data_in,
                target=target,
                result=result,
                loss_map=loss_map,
            )
        return loss

    def training_step(self, batch, batch_idx):
        log_row = batch_idx % self.train_log_row_interval == 0
        sample_name = ""

        loss = self.compute_loss(
            batch=batch, mode="train", log_row=log_row, sample_name=sample_name
        )
        return loss
