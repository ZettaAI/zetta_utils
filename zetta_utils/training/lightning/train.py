# pylint: disable=all # type: ignore
import os
from typing import Optional
import typeguard
import pytorch_lightning as pl

from zetta_utils import builder

builder.register("pl.Trainer")(pl.Trainer)
builder.register("pl.callbacks.ModelCheckpoint")(pl.callbacks.ModelCheckpoint)


@builder.register("lightning_train")
@typeguard.typechecked
def lighning_train(
    trainer,
    regime: pl.LightningModule,
    train_dataloader,
    val_dataloader=None,
    ckpt_path=None,
):
    trainer.fit(
        model=regime,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=ckpt_path,
    )
