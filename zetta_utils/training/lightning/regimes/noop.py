# pylint: disable=unused-argument
import time

import attrs
import pytorch_lightning as pl
import torch

from zetta_utils import builder

# import wandb

# from zetta_utils import viz


@builder.register("NoOpRegime")
@attrs.mutable(eq=False)
class NoOpRegime(pl.LightningModule):  # pylint: disable=too-many-ancestors
    def __attrs_pre_init__(self):
        super().__init__()

    def validation_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
        return None

    def training_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
        time.sleep(0.1)
        # self.log('yo', torch.tensor(0.45), on_step=True, on_epoch=True)
        # viz.rendering.Renderer()(torch.ones((1024, 1024)))
        # wandb.log(
        #    {
        #        "yo": wandb.Image(np.ones(1024, 1024))
        #    }
        # )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([torch.nn.Parameter()], lr=0)
        return optimizer
