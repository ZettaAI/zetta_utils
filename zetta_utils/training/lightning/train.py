import typeguard
import pytorch_lightning as pl

from zetta_utils import builder
from .regimes import TrainingRegime

builder.register("LightningTrainer")(pl.Trainer)


@builder.register("lightning_train")
@typeguard.typechecked
def train(trainer, regime: TrainingRegime, train_dataloader, val_dataloader=None, ckpt_path=None):
    """trainer = pl.Trainer(
        callbacks=callbacks,
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=log_every_n_steps,
        max_epochs=max_epochs,
        default_root_dir=root_dir,
    )"""

    trainer.fit(
        model=regime,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=ckpt_path,
    )
