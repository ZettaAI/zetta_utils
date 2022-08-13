from typing import Optional, List
import os
import json
import datetime
import pytorch_lightning as pl
import typeguard
import wandb
from zetta_utils import builder


@builder.register("ZettaDefaultTrainer")
@typeguard.typechecked
def build_default_trainer(
    experiment_name: str,
    experiment_version: str,
    checkpointing_kwargs: Optional[dict] = None,
    progress_bar_kwargs: Optional[dict] = None,
    **kwargs,
) -> pl.Trainer:
    if not os.environ.get("WANDB_MODE", None) == "offline":
        wandb.login()  # pragma: no cover
    logger = pl.loggers.WandbLogger(
        project=experiment_name,
        name=experiment_version,
        id=experiment_version,
    )
    if "ZETTA_RUN_SPEC" in os.environ:
        logger.experiment.config["zetta_run_spec"] = json.loads(os.environ["ZETTA_RUN_SPEC"])

    # Progress bar needs to be appended first to avoid default TQDM
    # bar being appended
    if progress_bar_kwargs is None:
        progress_bar_kwargs = {}
    prog_bar_callbacks = get_progress_bar_callbacks(
        **progress_bar_kwargs,
    )
    assert "callbacks" not in kwargs
    trainer = pl.Trainer(callbacks=prog_bar_callbacks, logger=logger, **kwargs)

    # Checkpoint callbacks need `default_root_dir`, so they're created
    # after
    if checkpointing_kwargs is None:
        checkpointing_kwargs = {}
    trainer.callbacks += get_checkpointing_callbacks(
        log_dir=os.path.join(trainer.default_root_dir, experiment_name, experiment_version),
        **checkpointing_kwargs,
    )

    return trainer


@typeguard.typechecked
def get_checkpointing_callbacks(
    log_dir: str,
    backup_every_n_secs: int = 900,
    update_every_n_secs: int = 60,
) -> List[pl.callbacks.Callback]:  # pragma: no cover
    result = [
        pl.callbacks.ModelCheckpoint(
            dirpath=log_dir,
            train_time_interval=datetime.timedelta(seconds=backup_every_n_secs),
            save_top_k=-1,
            save_last=False,
            monitor=None,
            filename="{epoch}-{step}-backup",
        ),
        pl.callbacks.ModelCheckpoint(
            dirpath=log_dir,
            train_time_interval=datetime.timedelta(seconds=update_every_n_secs),
            save_top_k=1,
            save_last=True,
            monitor=None,
            filename="{epoch}-{step}-current",
        ),
    ]  # type: List[pl.callbacks.Callback]
    return result


@typeguard.typechecked
def get_progress_bar_callbacks() -> List[pl.callbacks.Callback]:  # pragma: no cover
    result = [pl.callbacks.RichProgressBar()]  # type: List[pl.callbacks.Callback]
    return result
