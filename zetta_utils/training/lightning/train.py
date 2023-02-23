from __future__ import annotations

import json
import os

import pytorch_lightning as pl
import torch
import typeguard
from pytorch_lightning.utilities.cloud_io import get_filesystem

from zetta_utils import builder, log

logger = log.get_logger("zetta_utils")

builder.register("pl.Trainer")(pl.Trainer)
builder.register("pl.callbacks.ModelCheckpoint")(pl.callbacks.ModelCheckpoint)


@builder.register("lightning_train")
@typeguard.typechecked
def lightning_train(
    regime: pl.LightningModule,
    trainer: pl.Trainer,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader | None = None,
    full_state_ckpt_path: str = "last",
):
    """
    Perform neural net trainig with Zetta's PytorchLightning integration.

    :param regime: Training regime. Defines behavior on training, vallidation steps
    and epochs. Includes the model being trained as an instance variable.
    :param trainer: Pytorch Lightning Trainer object responsible for handling
        traing loop details that are common for all regimes, such as checkpointing
        behavior, logging behavior, etc. For Zetta training configuration, use
        ``zetta_utils.training.lightning.trainers.build_default_trainer``.
    :param train_dataloader: Training dataloader.
    :param val_dataloader: Validation dataloader.
    :param full_state_ckpt_path: Path to the training checkpoint to resume from.
        Must be a full training state checkpoint created by PytorchLightning rather
        than a model checkpoint. If ``full_state_ckpt_path=="last"``, the latest
        checkpoint for the given experiment will be identified and loaded.
    """
    logger.info("Starting training...")
    if "CURRENT_BUILD_SPEC" in os.environ:
        if hasattr(trainer, "log_config"):
            trainer.log_config(json.loads(os.environ["CURRENT_BUILD_SPEC"]))
        else:
            logger.warning("Incompatible custom trainer used: Unable to save configuration.")
    else:
        logger.warning("Invoked without builder: Unable to save configuration.")

    if full_state_ckpt_path == "last":
        if get_filesystem(trainer.ckpt_path).exists(trainer.ckpt_path):  # type: ignore
            ckpt_path = trainer.ckpt_path
        else:
            ckpt_path = None
    trainer.fit(
        model=regime,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=ckpt_path,
    )
