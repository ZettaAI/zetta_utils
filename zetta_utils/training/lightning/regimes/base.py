from abc import ABC, abstractmethod

import attrs
import torch
import pytorch_lightning as pl


@attrs.mutable
class TrainingRegime(pl.LightningModule, ABC):  # pylint: disable=too-many-ancestors
    lr: float

    @abstractmethod
    def save_model(self, path):
        """
        Defines how to save the **model** artifacts to the given path.
        This should not include optimizer parameters or any other Pytorch Lightning
        artifacts.
        """

    def configure_optimizers(self):
        """Override for using other optimizers."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
