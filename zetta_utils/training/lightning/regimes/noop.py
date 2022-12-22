# pylint: disable=unused-argument
import time

import attrs
import pytorch_lightning as pl
import torch

from zetta_utils import builder


@builder.register("NoOpRegime")
@attrs.mutable(eq=False)
class NoOpRegime(pl.LightningModule):  # pylint: disable=too-many-ancestors
    def __attrs_pre_init__(self):
        super().__init__()

    def validation_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
        return None

    def training_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
        time.sleep(0.2)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([torch.nn.Parameter()], lr=0)
        return optimizer
