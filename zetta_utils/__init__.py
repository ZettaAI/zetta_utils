# pylint: disable=unused-import, import-outside-toplevel, broad-exception-caught, import-error
"""Zetta AI Computational Connectomics Toolkit."""
import multiprocessing
import os
import sys
import threading
import time
import warnings
from typing import Literal

from .log import get_logger

# Set global multiprocessing context and threshold
MULTIPROCESSING_CONTEXT = "forkserver"
MULTIPROCESSING_NUM_TASKS_THRESHOLD = 128

# Forkserver initialization
LoadMode = Literal["all", "inference", "training", "try"]

_PRELOAD_MODULES: dict[LoadMode, str] = {
    "all": "zetta_utils.builder.preload.all",
    "inference": "zetta_utils.builder.preload.inference",
    "training": "zetta_utils.builder.preload.training",
    "try": "zetta_utils.builder.preload.try_load",
}

# Set start method to `forkserver` if not set elsewhere
# If not set here, `get_start_method` will set the default
# to `fork` w/o allow_none and cause issues with dependencies.
if multiprocessing.get_start_method(allow_none=True) is None:
    multiprocessing.set_start_method(MULTIPROCESSING_CONTEXT)


if "sphinx" not in sys.modules:  # pragma: no cover
    import pdbp  # noqa

    os.environ["PYTHONBREAKPOINT"] = "pdbp.set_trace"

logger = get_logger("zetta_utils")
ignore_warnings_from = [
    "python_jsonschema_objects",
    "kornia",
    "google",
    "pytorch_lightning",
    "lightning_fabric",
    "pkg_resources",
]

for pkg_name in ignore_warnings_from:
    warnings.filterwarnings("ignore", module=pkg_name)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def load_all_modules():  # pragma: no cover
    import zetta_utils.builder.preload.all


def try_load_train_inference():  # pragma: no cover
    import zetta_utils.builder.preload.try_load


def load_submodules():  # pragma: no cover
    from . import internal


def load_inference_modules():  # pragma: no cover
    import zetta_utils.builder.preload.inference


def load_training_modules():  # pragma: no cover
    import zetta_utils.builder.preload.training


def _noop() -> None:
    pass


def get_mp_context() -> multiprocessing.context.BaseContext:
    """Get the multiprocessing context for the configured start method."""
    return multiprocessing.get_context(MULTIPROCESSING_CONTEXT)


def initialize_forkserver(load_mode: LoadMode = "all") -> None:
    """Initialize forkserver with preloaded modules for the given load mode."""
    preload_module = _PRELOAD_MODULES[load_mode]
    logger.info(f"Configuring forkserver with preload module: {preload_module}")

    total_start = time.perf_counter()
    multiprocessing.set_forkserver_preload([preload_module])
    ctx = get_mp_context()
    proc = ctx.Process(target=_noop)  # type: ignore[attr-defined]
    proc.start()
    proc.join()

    total_elapsed = time.perf_counter() - total_start
    logger.info(f"Forkserver initialized in {total_elapsed:.2f}s (mode: {load_mode})")


def setup_environment(load_mode: LoadMode = "all") -> None:
    """
    Initialize forkserver and load modules in parallel.

    This function:
    1. Starts forkserver initialization in a background thread
    2. Loads modules in the main process (runs in parallel with forkserver init)
    3. Waits for forkserver to be ready before returning

    Args:
        load_mode: Which modules to load ("all", "inference", "training", "try")
    """
    # Start forkserver init in background while main process loads modules
    forkserver_thread = threading.Thread(
        target=initialize_forkserver, args=(load_mode,), name="forkserver_init"
    )
    forkserver_thread.start()

    # Load modules in main process (runs in parallel with forkserver init)
    if load_mode == "all":
        load_all_modules()
    elif load_mode == "inference":  # pragma: no cover
        load_inference_modules()
    elif load_mode == "try":  # pragma: no cover
        try_load_train_inference()
    else:  # training  # pragma: no cover
        load_training_modules()

    # Wait for forkserver to be ready before proceeding
    forkserver_thread.join()
