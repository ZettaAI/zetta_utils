from __future__ import annotations

import datetime
import importlib.metadata
import json
import os
from typing import Any, Dict, List, Optional

import fsspec
import pytorch_lightning as pl
import torch
import typeguard
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import ddp

from zetta_utils import builder, log

logger = log.get_logger("zetta_utils")
ONNX_OPSET_VERSION = 17
os.environ["MKL_THREADING_LAYER"] = "GNU"


"""
Separate function to work around the jit.trace memory leak
"""


def trace_and_save_model(
    args_packed,
):  # pragma: no cover # pylint: disable=broad-except, used-before-assignment
    model, trace_input, filepath, name = args_packed
    trace = torch.jit.trace(model, trace_input)
    filepath_jit = f"{filepath}.static-{torch.__version__}-{name}.jit"
    with fsspec.open(filepath_jit, "wb") as f:
        torch.jit.save(trace, f)
    try:
        filepath_onnx = f"{filepath}.static-{torch.__version__}-{name}.onnx"
        with fsspec.open(filepath_onnx, "wb") as f:
            filesystem = f.fs
            torch.onnx.export(model, trace_input, f, opset_version=ONNX_OPSET_VERSION)
        return None
    except Exception as e:
        filesystem.delete(filepath_onnx)
        return type(e).__name__, e.args[0]


class ZettaDefaultTrainer(pl.Trainer):  # pragma: no cover
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace_configuration = None

    # TODO: For PL>=1,8.0, the line should be @pl.utilities.rank_zero.rank_zero_only
    @pl.utilities.distributed.rank_zero_only
    def save_checkpoint(
        self, filepath, weights_only: bool = False, storage_options: Optional[Any] = None
    ):  # pylint: disable=too-many-locals

        if filepath.startswith("./"):
            filepath = f"{self.default_root_dir}/{filepath[2:]}"
        super().save_checkpoint(filepath, weights_only, storage_options)

        regime = self.lightning_module
        for k, v in regime._modules.items():  # pylint: disable=protected-access
            if hasattr(v, "__built_with_spec"):
                model_spec = getattr(v, "__built_with_spec")  # pylint: disable=protected-access
                while "@type" in model_spec and model_spec["@type"] == "load_weights_file":
                    model_spec = model_spec["model"]

                spec = {
                    "@type": "load_weights_file",
                    "@version": importlib.metadata.version("zetta_utils"),
                    "model": model_spec,
                    "ckpt_path": filepath,
                    "component_names": [k],
                    "remove_component_prefix": True,
                    "strict": True,
                }
                spec_path = f"{filepath}.{k}.spec.json"
                with fsspec.open(spec_path, "w") as f:
                    json.dump(spec, f, indent=3)

        if self.trace_configuration is not None:
            for name in self.trace_configuration.keys():
                model = self.trace_configuration[name]["model"]
                trace_input = self.trace_configuration[name]["trace_input"]
                ctx = torch.multiprocessing.get_context("spawn")
                with ctx.Pool(processes=1) as pool:
                    res = pool.map(trace_and_save_model, [(model, trace_input, filepath, name)])[0]
                if res is not None:
                    logger.warning(f"Exception while saving the model as ONNX: {res[0]}: {res[1]}")


@builder.register("ZettaDefaultTrainer")
@typeguard.typechecked
def build_default_trainer(
    experiment_name: str,
    experiment_version: str,
    checkpointing_kwargs: Optional[dict] = None,
    progress_bar_kwargs: Optional[dict] = None,
    **kwargs,
) -> pl.Trainer:
    if not os.environ.get("WANDB_MODE", None) == "offline":  # pragma: no cover
        api_key = os.environ.get("WANDB_API_KEY", None)
        wandb.login(key=api_key)
        wandb.init(group=f"{experiment_name}.{experiment_version}")

    wandb_logger = WandbLogger(
        project=experiment_name,
        name=experiment_version,
        id=experiment_version,
    )

    if wandb.run is not None:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        zetta_root_path = f"{this_dir}/../../.."
        wandb.run.log_code(zetta_root_path)

    # Progress bar needs to be appended first to avoid default TQDM
    # bar being appended
    if progress_bar_kwargs is None:
        progress_bar_kwargs = {}
    prog_bar_callbacks = get_progress_bar_callbacks(
        **progress_bar_kwargs,
    )
    assert "callbacks" not in kwargs

    if "strategy" not in kwargs:
        kwargs["strategy"] = ddp.DDPStrategy(find_unused_parameters=False)
    trainer = ZettaDefaultTrainer(callbacks=prog_bar_callbacks, logger=wandb_logger, **kwargs)

    # Checkpoint callbacks need `default_root_dir`, so they're created
    # after
    if checkpointing_kwargs is None:
        checkpointing_kwargs = {}
    log_dir = os.path.join(
        trainer.default_root_dir,
        experiment_name,
        experiment_version,
    )

    trainer.callbacks += get_checkpointing_callbacks(
        log_dir=log_dir,
        **checkpointing_kwargs,
    )

    trainer.callbacks.append(ConfigureTraceCallback(trainer))

    # Due to a bug in PL we're unable to use normal methods
    # to resume training with ckpt_path='last' when storing
    # checkpoints on GCP.
    trainer._ckpt_path = os.path.join(log_dir, "last.ckpt")  # pylint: disable=protected-access

    # def log_config(config):
    #     if experiment_version.startswith("tmp"):
    #         logger.info(
    #             f"Not saving configuration for a temproary experiment {experiment_version}."
    #         )
    #     else:
    #         wandb_logger.experiment.config["training_configuration"] = config
    #         logger.info("Saved training configuration.")

    # trainer.log_config = log_config  # type: ignore # pylint: disable=attribute-defined-outside-init

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


@typeguard.typechecked
class ConfigureTraceCallback(pl.callbacks.Callback):  # pragma: no cover
    def __init__(self, trainer: pl.Trainer) -> None:
        self.trainer = trainer
        self.pl_module = trainer.lightning_module

    @staticmethod
    def wrap_forward(
        pl_module: pl.LightningModule, name: str, trace_configuration: Dict[str, Dict[str, Any]]
    ) -> None:
        model = getattr(pl_module, name)
        model.__forward__ = model.forward

        def wrapped_forward(*args, **kwargs):
            trace_configuration[name] = {"model": model, "trace_input": args}
            return model.__forward__(*args, **kwargs)

        setattr(model, "forward", wrapped_forward)

    @staticmethod
    def unwrap_forward(pl_module: pl.LightningModule, name: str) -> None:
        model = getattr(pl_module, name)
        wrapped_forward = model.forward
        model.forward = model.__forward__
        delattr(model, "__forward__")
        del wrapped_forward

    def on_validation_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if (
            hasattr(self.trainer, "trace_configuration")
            and self.trainer.trace_configuration is None
        ):
            input_to_trace = batch
            models_to_trace = [
                attr
                for attr in set(dir(pl_module))
                if issubclass(type(getattr(pl_module, attr)), torch.nn.Module)
            ]
            trace_configuration = {}  # type: Dict[str, Dict[str, Any]]
            for name in models_to_trace:
                ConfigureTraceCallback.wrap_forward(pl_module, name, trace_configuration)
            pl_module.validation_step(input_to_trace, 0)
            for name in models_to_trace:
                ConfigureTraceCallback.unwrap_forward(pl_module, name)
            self.trainer.trace_configuration = trace_configuration
