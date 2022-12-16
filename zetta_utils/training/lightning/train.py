import json
import os

import pytorch_lightning as pl
import typeguard
from pytorch_lightning.utilities.cloud_io import get_filesystem

from zetta_utils import builder, log

logger = log.get_logger("zetta_utils")

builder.register("pl.Trainer")(pl.Trainer)
builder.register("pl.callbacks.ModelCheckpoint")(pl.callbacks.ModelCheckpoint)


@builder.register("lightning_train")
@typeguard.typechecked
def lightning_train(
    trainer,
    regime: pl.LightningModule,
    train_dataloader,
    val_dataloader=None,
    full_state_ckpt_path="last",
):
    logger.info("Starting training...")
    if "CURRENT_BUILD_SPEC" in os.environ:
        if hasattr(trainer, "log_config"):
            trainer.log_config(json.loads(os.environ["CURRENT_BUILD_SPEC"]))
            logger.info("Saved training configuration.")
        else:
            logger.warning("Incompatible custom trainer used: Unable to save configuration.")
    else:
        logger.warning("Invoked without builder: Unable to save configuration.")

    if full_state_ckpt_path == "last":
        if get_filesystem(trainer.ckpt_path).exists(trainer.ckpt_path):
            ckpt_path = trainer.ckpt_path
        else:
            ckpt_path = None
    trainer.fit(
        model=regime,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=ckpt_path,
    )
