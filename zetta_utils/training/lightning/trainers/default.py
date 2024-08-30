from __future__ import annotations

import datetime
import importlib.metadata
import os
import time
from typing import Any, Dict, List, Optional

import fsspec
import pytorch_lightning as pl
import torch
import typeguard
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import ddp

from zetta_utils import builder, log
from zetta_utils.builder import get_initial_builder_spec
from zetta_utils.parsing import json
from zetta_utils.typing import JsonSerializableValue

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
    # trace = torch.jit.trace(model, trace_input)
    # filepath_jit = f"{filepath}.static-{torch.__version__}-{name}.jit"
    # with fsspec.open(filepath_jit, "wb") as f:
    #     torch.jit.save(trace, f)
    # try:
    #     filepath_onnx = f"{filepath}.static-{torch.__version__}-{name}.onnx"
    #     with fsspec.open(filepath_onnx, "wb") as f:
    #         filesystem = f.fs
    #         torch.onnx.export(model, trace_input, f, opset_version=ONNX_OPSET_VERSION)
    #     return None
    # except Exception as e:
    #     filesystem.delete(filepath_onnx)
    #     return type(e).__name__, e.args[0]


@builder.register("ZettaDefaultTrainer")
class ZettaDefaultTrainer(pl.Trainer):  # pragma: no cover
    def __init__(
        self,
        experiment_name: str,
        experiment_version: str,
        *args,
        checkpointing_kwargs: Optional[dict] = None,
        progress_bar_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        assert "callbacks" not in kwargs
        if "strategy" not in kwargs:
            kwargs["strategy"] = ddp.DDPStrategy(find_unused_parameters=False)

        if checkpointing_kwargs is None:
            checkpointing_kwargs = {}

        if progress_bar_kwargs is None:
            progress_bar_kwargs = {}

        kwargs["logger"] = False

        log_dir = os.path.join(
            kwargs.get("default_root_dir", os.getcwd()),
            experiment_name,
            f"{experiment_version}",
        )

        kwargs["callbacks"] = get_progress_bar_callbacks(
            **progress_bar_kwargs
        ) + get_checkpointing_callbacks(
            log_dir=log_dir,
            **checkpointing_kwargs,
        )

        super().__init__(*args, **kwargs)

        self.trace_configuration: Dict = {}

        # self.callbacks will exist at runtime
        self.callbacks.append(ConfigureTraceCallback(self))  # type: ignore
        self.callbacks.append(  # type: ignore
            ConfigureLogging(experiment_name, experiment_version)
        )
        self.callbacks.append(WallClockTimeCallback())  # type: ignore

        # Due to a bug in PL we're unable to use normal methods
        # to resume training with ckpt_path='last' when storing
        # checkpoints on GCP.
        self._ckpt_path = os.path.join(log_dir, "last.ckpt")

    @pl.utilities.rank_zero.rank_zero_only
    def save_checkpoint(
        self, filepath, weights_only: bool = False, storage_options: Optional[Any] = None
    ):  # pylint: disable=too-many-locals
        if filepath.startswith("./"):
            filepath = f"{self.default_root_dir}/{filepath[2:]}"
        super().save_checkpoint(filepath, weights_only, storage_options)

        regime = self.lightning_module
        for k, v in regime._modules.items():  # pylint: disable=protected-access
            model_spec: JsonSerializableValue = get_initial_builder_spec(v)
            if model_spec is not None:
                unrolled_spec: JsonSerializableValue = model_spec
                while (
                    isinstance(unrolled_spec, dict)
                    and "@type" in unrolled_spec
                    and unrolled_spec["@type"] == "load_weights_file"
                ):
                    unrolled_spec = unrolled_spec["model"]

                spec = {
                    "@type": "load_weights_file",
                    "@version": importlib.metadata.version("zetta_utils"),
                    "model": unrolled_spec,
                    "ckpt_path": filepath,
                    "component_names": [k],
                    "remove_component_prefix": True,
                    "strict": True,
                }
                spec_path = f"{filepath}.{k}.spec.json"
                with fsspec.open(spec_path, "w") as f:
                    json.dump(spec, f, indent=3)

        for name, val in self.trace_configuration.items():
            model = val["model"]
            trace_input = val["trace_input"]
            ctx = torch.multiprocessing.get_context("spawn")
            with ctx.Pool(processes=1) as pool:
                # See https://github.com/pytorch/pytorch/issues/35600
                res = pool.map(trace_and_save_model, [(model, trace_input, filepath, name)])[0]
            if res is not None:
                logger.warning(f"Exception while saving the model as ONNX: {res[0]}: {res[1]}")


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
def get_progress_bar_callbacks(**kwargs) -> List[pl.callbacks.Callback]:  # pragma: no cover
    result = [pl.callbacks.RichProgressBar(**kwargs)]  # type: List[pl.callbacks.Callback]
    return result


@typeguard.typechecked
class ConfigureTraceCallback(pl.callbacks.Callback):  # pragma: no cover
    def __init__(self, trainer: pl.Trainer) -> None:
        self.trainer = trainer
        self.pl_module = trainer.lightning_module

    @staticmethod
    def wrap_forward(pl_module: pl.LightningModule, name: str, trace_configuration: Dict) -> None:
        model = pl_module.get_submodule(name)
        setattr(model, "__forward__", model.forward)

        def wrapped_forward(*args, **kwargs):
            trace_configuration[name] = {"model": model, "trace_input": args}
            return getattr(model, "__forward__")(*args, **kwargs)

        setattr(model, "forward", wrapped_forward)

    @staticmethod
    def unwrap_forward(pl_module: pl.LightningModule, name: str) -> None:
        model = pl_module.get_submodule(name)
        wrapped_forward = model.forward
        model.forward = getattr(model, "__forward__")
        delattr(model, "__forward__")
        del wrapped_forward

    def on_validation_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if hasattr(self.trainer, "trace_configuration") and not self.trainer.trace_configuration:
            input_to_trace = batch
            models_to_trace = [name for name, _ in pl_module.named_children()]
            trace_configuration = {}  # type: Dict[str, Dict[str, Any]]
            for name in models_to_trace:
                ConfigureTraceCallback.wrap_forward(pl_module, name, trace_configuration)
            pl_module.validation_step(input_to_trace, 0)
            for name in models_to_trace:
                ConfigureTraceCallback.unwrap_forward(pl_module, name)
            self.trainer.trace_configuration = trace_configuration


class ConfigureLogging(pl.callbacks.Callback):
    def __init__(
        self,
        exp_name: str,
        exp_version: str,
    ) -> None:
        self.exp_name = exp_name
        self.exp_version = exp_version

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not os.environ.get("WANDB_MODE", None) == "offline":  # pragma: no cover
            api_key = os.environ.get("WANDB_API_KEY", None)
            wandb.login(key=api_key)

        trainer.logger = WandbLogger(
            project=self.exp_name,
            group=f"{self.exp_name}.{self.exp_version}",
            name=f"{self.exp_version}",
            id=f"{self.exp_version}.{trainer.global_rank}",
        )

        if trainer.global_rank != 0:
            return

        if trainer.logger and trainer.logger.experiment:
            this_dir = os.path.dirname(os.path.abspath(__file__))
            zetta_root_path = f"{this_dir}/../../.."
            trainer.logger.experiment.log_code(zetta_root_path)

        def log_config(config):
            if self.exp_version.startswith("tmp"):
                logger.info(f"Not saving configuration for a temp experiment {self.exp_version}.")
            else:
                self.logger.experiment.config["training_configuration"] = config  # type: ignore
                logger.info("Saved training configuration.")

        trainer.log_config = log_config  # type: ignore


@typeguard.typechecked
class WallClockTimeCallback(pl.callbacks.Callback):  # pragma: no cover
    def __init__(self) -> None:
        super().__init__()
        self.start_time = 0.0

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.start_time = time.perf_counter()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: torch.Tensor | dict[str, Any],
        batch: Any,
        batch_idx: int,
    ) -> None:
        end_time = time.perf_counter()
        elapsed_time = end_time - self.start_time
        pl_module.log("elapsed/train", elapsed_time, on_step=True, on_epoch=True)
