import attrs
import torch
import pytorch_lightning as pl

import zetta_utils as zu


from zetta_utils import builder, convnet  # pylint: disable=unused-import


@builder.register("NaiveSupervised")
@attrs.mutable(eq=False)
class NaiveSupervised(pl.LightningModule):  # pylint: disable=too-many-ancestors
    model: torch.nn.Module
    lr: float

    def __attrs_pre_init__(self):
        super().__init__()

    def save_model(self, path: str):
        zu.convnet.utils.save_model(self.model, path)

    def training_step(self, batch, _):  # pylint: disable=arguments-differ
        data_in = batch["data_in"]
        target = batch["target"]

        if (data_in != 0).sum() == 0:
            loss = None
        else:
            result = self.model(data_in)
            loss_map = (target - result) ** 2
            if "loss_weights" in batch:
                loss_weights = batch["loss_weights"]
                loss = (loss_map * loss_weights).sum()
            else:
                loss = loss_map.sum()
            self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        """Override for using other optimizers."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
