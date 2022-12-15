import pytorch_lightning as pl
import typeguard
from pytorch_lightning.utilities.cloud_io import get_filesystem

from zetta_utils import builder

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
