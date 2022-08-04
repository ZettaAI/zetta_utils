from typing import Optional
import pytorch_lightning as pl
import typeguard
from zetta_utils import builder


@builder.register("ZettaDefaultTrainer")
@typeguard.typechecked
def build_default_trainer(checkpointing_kwargs: Optional[dict] = None, **kwargs) -> pl.Trainer:
    trainer = pl.Trainer(**kwargs)
    if checkpointing_kwargs is None:
        checkpointing_kwargs = {}
    add_checkpointing_callbacks(trainer, **checkpointing_kwargs)
    return trainer


@typeguard.typechecked
def add_checkpointing_callbacks(trainer: pl.Trainer, every_n_steps: int = 100):
    trainer.callbacks.append(
        pl.callbacks.ModelCheckpoint(
            every_n_train_steps=every_n_steps,
            save_top_k=3,
            save_last=True,
            monitor="train_loss",
        )
    )
    trainer.callbacks.append(
        pl.callbacks.ModelCheckpoint(
            every_n_epochs=1,
            save_top_k=-1,
        )
    )

    return trainer
