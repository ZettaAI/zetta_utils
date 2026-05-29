from __future__ import annotations

import datetime
import importlib.metadata
import os
import shutil
import signal
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import fsspec
import lightning.pytorch as pl
import torch
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as _mp
import typeguard
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.plugins.io import TorchCheckpointIO
from lightning.pytorch.strategies import ddp
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.format_utils import _EmptyStateDictLoadPlanner
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
from torch.distributed.checkpoint.state_dict_saver import AsyncCheckpointerType

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


_mp_ctx = _mp.get_context("spawn")


def _stage_checkpoint(data):
    """Deep-copy checkpoint, placing tensors in shared memory."""
    if isinstance(data, torch.Tensor):
        return data.clone().share_memory_()
    if isinstance(data, dict):
        return {k: _stage_checkpoint(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(_stage_checkpoint(v) for v in data)
    return data


def _save_checkpoint_worker(checkpoint, path, storage_options, error_queue):
    """Runs in a separate process -- no GIL contention with training."""
    try:
        TorchCheckpointIO().save_checkpoint(checkpoint, path, storage_options)
    except Exception as e:  # pylint: disable=broad-exception-caught
        error_queue.put(f"{type(e).__name__}: {e}")


class AsyncCheckpointIO(TorchCheckpointIO):  # pragma: no cover
    """Process-based async checkpoint IO for single-node training.

    Avoids GIL contention by running torch.save + storage I/O in a
    separate process. Checkpoint tensors are staged to shared memory
    for efficient cross-process access.

    For distributed training (world_size > 1), ZettaDefaultTrainer uses
    torch.distributed.checkpoint (DCP) directly instead of this class.
    """

    def __init__(self) -> None:
        super().__init__()
        self._save_process: Any = None
        self._error_queue: _mp.Queue = _mp_ctx.Queue()

    def save_checkpoint(
        self, checkpoint: Dict[str, Any], path: Union[str, Path], storage_options=None
    ) -> None:
        self._wait_pending()
        staged = _stage_checkpoint(checkpoint)
        self._save_process = _mp_ctx.Process(
            target=_save_checkpoint_worker,
            args=(staged, path, storage_options, self._error_queue),
        )
        self._save_process.start()

    def load_checkpoint(
        self, path: Union[str, Path], map_location=None, weights_only=None
    ) -> Dict[str, Any]:
        if _is_dcp_checkpoint(str(path)):
            return _load_dcp_checkpoint(str(path), map_location)
        return super().load_checkpoint(path, map_location=map_location, weights_only=weights_only)

    def remove_checkpoint(  # pylint: disable=arguments-renamed
        self, filepath: Union[str, Path]
    ) -> None:
        filepath = str(filepath)
        if _is_dcp_checkpoint(filepath):
            if "://" in filepath:
                fs, _ = fsspec.core.url_to_fs(filepath)
                fs.rm(filepath, recursive=True)
            else:
                shutil.rmtree(filepath)
        else:
            super().remove_checkpoint(filepath)

    def _wait_pending(self) -> None:
        if self._save_process is not None:
            self._save_process.join(timeout=300)
            if self._save_process.is_alive():
                logger.warning("Async checkpoint save timed out, killing save process")
                self._save_process.kill()
                self._save_process.join()
            if self._save_process.exitcode != 0:
                err = ""
                if not self._error_queue.empty():
                    err = self._error_queue.get_nowait()
                raise RuntimeError(f"Async checkpoint save failed: {err}")
            self._save_process = None

    def teardown(self) -> None:
        self._wait_pending()


@builder.register("ZettaDefaultTrainer")
class ZettaDefaultTrainer(pl.Trainer):  # pragma: no cover
    def __init__(
        self,
        experiment_name: str,
        experiment_version: str,
        *args,
        checkpointing_kwargs: Optional[dict] = None,
        progress_bar_kwargs: Optional[dict] = None,
        async_checkpointing: bool = False,
        trace_exports: bool = True,
        static_graph: bool = False,
        ddp_init_timeout_minutes: float = 2,
        **kwargs,
    ):
        assert "callbacks" not in kwargs

        world_size = 1
        if "WORLD_SIZE" in os.environ and "LOCAL_WORLD_SIZE" in os.environ:
            local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
            world_size = int(os.environ["WORLD_SIZE"])
            kwargs["num_nodes"] = world_size // local_world_size
            kwargs["devices"] = local_world_size

        self._use_dcp = world_size > 1
        self._async_checkpointing = async_checkpointing

        if "strategy" not in kwargs:
            kwargs["strategy"] = ddp.DDPStrategy(
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
                static_graph=static_graph,
                timeout=datetime.timedelta(minutes=ddp_init_timeout_minutes),
                process_group_backend="cpu:gloo,cuda:nccl" if self._use_dcp else None,
            )

        if self._use_dcp:
            self._upload_thread = None
            self._upload_error_holder = None
            self._local_ckpt_tmpdir = tempfile.mkdtemp(prefix="dcp_ckpt_")

        if self._use_dcp or async_checkpointing:
            strategy = kwargs.get("strategy")
            if isinstance(strategy, ddp.DDPStrategy):
                strategy.checkpoint_io = AsyncCheckpointIO()
            else:
                kwargs.setdefault("plugins", [])
                kwargs["plugins"].append(AsyncCheckpointIO())

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

        kwargs["callbacks"] = get_checkpointing_callbacks(
            log_dir=log_dir,
            **checkpointing_kwargs,
        )
        if kwargs.setdefault("enable_progress_bar", True):
            kwargs["callbacks"].extend(get_progress_bar_callbacks(**progress_bar_kwargs))

        super().__init__(*args, **kwargs)

        self.trace_configuration: Dict = {}

        # self.callbacks will exist at runtime
        if trace_exports:
            self.callbacks.append(ConfigureTraceCallback(self))  # type: ignore
        self.callbacks.append(  # type: ignore
            ConfigureLogging(experiment_name, experiment_version)
        )
        self.callbacks.insert(0, WallClockTimeCallback())  # type: ignore
        self.callbacks.append(SIGTERMCheckpointCallback())  # type: ignore

        self._ckpt_path = os.path.join(log_dir, "last.ckpt")

    def fit(self, *args, **kwargs):
        try:
            super().fit(*args, **kwargs)
        finally:
            if self._use_dcp:
                self._dcp_wait_upload()
                shutil.rmtree(self._local_ckpt_tmpdir, ignore_errors=True)

    def save_checkpoint(
        self, filepath, weights_only: bool | None = False, storage_options: Any | None = None
    ):  # pylint: disable=too-many-locals
        if filepath.startswith("./"):
            filepath = f"{self.default_root_dir}/{filepath[2:]}"

        logger.info(
            f"Saving checkpoint to {filepath} "
            f"(epoch={self.current_epoch}, global_step={self.global_step})"
        )

        if self._use_dcp:
            checkpoint = self._checkpoint_connector.dump_checkpoint(weights_only)
            self._dcp_save(checkpoint, filepath)
        else:
            super().save_checkpoint(filepath, weights_only, storage_options)

        if not self.is_global_zero:
            return

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

    def _dcp_save(self, checkpoint: Dict[str, Any], filepath: str) -> None:
        self._dcp_wait_upload()

        local_dir = os.path.join(self._local_ckpt_tmpdir, os.path.basename(filepath))
        if os.path.exists(local_dir):
            shutil.rmtree(local_dir)

        if self._async_checkpointing:
            future = dcp.async_save(
                checkpoint,
                checkpoint_id=local_dir,
                async_checkpointer_type=AsyncCheckpointerType.PROCESS,
            )
            self._upload_error_holder = []
            self._upload_thread = threading.Thread(
                target=_wait_and_upload,
                args=(future, local_dir, filepath, self._upload_error_holder),
                daemon=True,
            )
            self._upload_thread.start()
        else:
            dcp.save(checkpoint, checkpoint_id=local_dir)
            _upload_checkpoint(local_dir, filepath)
            shutil.rmtree(local_dir, ignore_errors=True)

    def _dcp_wait_upload(self) -> None:
        if self._upload_thread is not None:
            self._upload_thread.join()
            self._upload_thread = None
            if self._upload_error_holder:
                err = self._upload_error_holder[0]
                self._upload_error_holder = None
                raise RuntimeError(f"DCP checkpoint upload failed: {err}") from err
            self._upload_error_holder = None


def _upload_checkpoint(local_dir: str, dest_path: str) -> None:
    """Upload a local DCP checkpoint directory to the final destination.

    Uses individual file puts to avoid gcsfs directory nesting bug (see pitfall #5).
    """
    if "://" in dest_path:
        fs, _ = fsspec.core.url_to_fs(dest_path)
        for fname in os.listdir(local_dir):
            fs.put(os.path.join(local_dir, fname), f"{dest_path}/{fname}")
    else:
        if os.path.exists(dest_path):
            shutil.rmtree(dest_path)
        shutil.copytree(local_dir, dest_path)


def _wait_and_upload(future, local_dir: str, dest_path: str, error_holder: list) -> None:
    """Wait for async DCP save to complete, then upload to final destination."""
    try:
        future.result()
        _upload_checkpoint(local_dir, dest_path)
        shutil.rmtree(local_dir, ignore_errors=True)
    except BaseException as e:  # pylint: disable=broad-exception-caught
        logger.error(f"DCP checkpoint upload failed: {e}")
        error_holder.append(e)


def _is_dcp_checkpoint(path: str) -> bool:
    """Check if the given path contains a DCP-format checkpoint."""
    metadata_path = os.path.join(path, ".metadata")
    if "://" in path:
        fs, _ = fsspec.core.url_to_fs(path)
        return fs.exists(metadata_path)
    return os.path.isdir(path) and os.path.exists(metadata_path)


def _fix_dcp_optimizer_keys(checkpoint: Dict[str, Any]) -> None:
    """Fix optimizer state dict keys after DCP load.

    DCP's flatten/unflatten converts int dict keys to strings (str(k) during
    flatten, string path elements during unflatten). PyTorch's
    Optimizer.load_state_dict expects int keys in 'state' to match the int
    param IDs from param_groups['params'].
    """
    for opt_state in checkpoint.get("optimizer_states", []):
        if not isinstance(opt_state, dict):
            continue
        if "state" in opt_state and isinstance(opt_state["state"], dict):
            opt_state["state"] = {
                int(k) if isinstance(k, str) and k.isdigit() else k: v
                for k, v in opt_state["state"].items()
            }


def _load_dcp_checkpoint(path: str, map_location=None) -> Dict[str, Any]:
    """Load a DCP-format checkpoint, downloading from remote if needed."""
    logger.info(f"Loading DCP checkpoint from {path}")

    if "://" in path:
        local_dir = tempfile.mkdtemp(prefix="dcp_load_")
        fs, _ = fsspec.core.url_to_fs(path)
        basename = os.path.basename(path.rstrip("/"))
        load_path = os.path.join(local_dir, basename)
        os.makedirs(load_path, exist_ok=True)
        for finfo in fs.ls(path, detail=False):
            fs.get(finfo, os.path.join(load_path, os.path.basename(finfo)))
    else:
        local_dir = None
        load_path = path

    try:
        checkpoint: Dict[str, Any] = {}
        _load_state_dict(
            checkpoint,
            storage_reader=FileSystemReader(load_path),
            planner=_EmptyStateDictLoadPlanner(),
            no_dist=True,
        )

        _fix_dcp_optimizer_keys(checkpoint)

        if map_location is not None:
            checkpoint = _apply_map_location(checkpoint, map_location)

        logger.info(
            f"DCP checkpoint loaded: epoch={checkpoint.get('epoch')}, "
            f"global_step={checkpoint.get('global_step')}"
        )
        return checkpoint
    finally:
        if local_dir is not None:
            shutil.rmtree(local_dir, ignore_errors=True)


def _apply_map_location(data: Any, map_location) -> Any:
    if isinstance(data, torch.Tensor):
        return data.to(map_location)
    if isinstance(data, dict):
        return {k: _apply_map_location(v, map_location) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(_apply_map_location(v, map_location) for v in data)
    return data


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

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):  # pragma: no cover
        if not os.environ.get("WANDB_MODE", None) == "offline":
            api_key = os.environ.get("WANDB_API_KEY", None)
            wandb.login(key=api_key)

        wandb.finish(quiet=True)

        trainer.logger = WandbLogger(
            project=self.exp_name,
            group=f"{self.exp_name}.{self.exp_version}",
            name=f"{self.exp_version}",
            id=f"{self.exp_version}.{trainer.global_rank}",
        )

        if trainer.global_rank != 0:
            return

        current_build_spec = os.environ.pop("CURRENT_BUILD_SPEC", None)
        try:
            experiment = trainer.logger.experiment if trainer.logger else None
            if experiment:
                this_dir = os.path.dirname(os.path.abspath(__file__))
                zetta_root_path = f"{this_dir}/../../.."
                experiment.log_code(zetta_root_path)
        finally:
            if current_build_spec is not None:
                os.environ["CURRENT_BUILD_SPEC"] = current_build_spec

        def log_config(config):
            if self.exp_version.startswith("tmp"):
                logger.info(f"Not saving configuration for a temp experiment {self.exp_version}.")
            else:
                trainer.logger.experiment.config["training_configuration"] = config  # type: ignore
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
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        end_time = time.perf_counter()
        elapsed_time = end_time - self.start_time
        pl_module.log("elapsed/train", elapsed_time, on_step=True)


class SIGTERMCheckpointCallback(pl.callbacks.Callback):  # pragma: no cover
    """Saves an emergency checkpoint when SIGTERM is received (e.g., spot preemption).

    Installs a SIGTERM handler at fit start that sets a flag. At the end of each
    training batch, if the flag is set, triggers an immediate checkpoint save and
    raises ``SystemExit`` to stop training gracefully.
    """

    def __init__(self) -> None:
        super().__init__()
        self._sigterm_received = False
        self._original_handler: Union[Callable[..., Any], int, None] = None

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._original_handler = signal.getsignal(signal.SIGTERM)

        def _handler(signum, frame):  # pylint: disable=unused-argument
            logger.warning("SIGTERM received — will exit after current step.")
            self._sigterm_received = True

        signal.signal(signal.SIGTERM, _handler)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._original_handler is not None:
            signal.signal(signal.SIGTERM, self._original_handler)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self._sigterm_received:
            logger.warning("SIGTERM received, exiting training.")
            if self._original_handler is not None:
                signal.signal(signal.SIGTERM, self._original_handler)
            raise SystemExit("SIGTERM received, exiting.")
