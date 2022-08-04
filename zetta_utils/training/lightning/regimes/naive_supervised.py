import torch
import attrs

import zetta_utils as zu

from zetta_utils import builder, convnet  # pylint: disable=unused-import
from zetta_utils.training.lightning.regimes.base import TrainingRegime


@builder.register("NaiveSupervised")
@attrs.mutable
class NaiveSupervised(TrainingRegime):  # pylint: disable=too-many-ancestors
    model: torch.nn.Module

    def save_model(self, path: str):
        zu.convnet.utils.save_model(self.model, path)

    def train_step(self, batch, _):
        data_in = batch["data_in"]
        target = batch["target"]

        result = self.model(data_in)
        loss_map = (target - result) ** 2
        if "loss_weights" in batch:
            loss_weights = batch["loss_weights"]
            loss = (loss_map * loss_weights).sum()
        else:
            loss = loss_map
        self.log("train_loss", loss)
        return loss
