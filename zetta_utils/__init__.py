# pylint: disable=unused-import, import-outside-toplevel, broad-exception-caught, import-error
"""Zetta AI Computational Connectomics Toolkit."""
import faulthandler
import multiprocessing
import os
import sys
import time
import warnings
from typing import Literal

faulthandler.enable(all_threads=True)

from .log import get_logger
from .parallel import get_mp_context  # noqa: F401


def _patch_gcsfs_for_proxy():
    """Patch gcsfs to respect HTTP_PROXY/HTTPS_PROXY environment variables.

    By default, aiohttp (used by gcsfs) ignores proxy env vars.
    This patch injects `session_kwargs={'trust_env': True}` into all
    GCSFileSystem instances so they automatically use HTTP_PROXY/HTTPS_PROXY.
    """
    try:
        import gcsfs

        _original_init = gcsfs.GCSFileSystem.__init__

        def _patched_init(self, *args, **kwargs):
            session_kwargs = kwargs.get("session_kwargs", {})
            session_kwargs.setdefault("trust_env", True)
            kwargs["session_kwargs"] = session_kwargs
            return _original_init(self, *args, **kwargs)

        gcsfs.GCSFileSystem.__init__ = _patched_init
    except ImportError:  # pragma: no cover
        pass


_patch_gcsfs_for_proxy()


# Forkserver initialization
LoadMode = Literal["all", "inference", "training", "try"]

_PRELOAD_MODULES: dict[LoadMode, str] = {
    "all": "zetta_utils.builder.preload.all",
    "inference": "zetta_utils.builder.preload.inference",
    "training": "zetta_utils.builder.preload.training",
    "try": "zetta_utils.builder.preload.try_load",
}


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


def _spawn_forkserver_daemon(load_mode: LoadMode) -> None:
    """Spawn the daemon (fork+exec); return before its preload imports finish."""
    # pylint: disable=import-outside-toplevel,protected-access
    from multiprocessing.forkserver import _forkserver

    multiprocessing.set_forkserver_preload([_PRELOAD_MODULES[load_mode]])
    _forkserver.ensure_running()


def _wait_for_forkserver_ready() -> None:
    """Block until the daemon can serve fork requests."""
    ctx = get_mp_context()
    proc = ctx.Process(target=_noop)
    proc.start()
    proc.join()


def _inherited_forkserver_daemon() -> bool:
    """True if we inherited a running forkserver daemon from our parent."""
    # pylint: disable=import-outside-toplevel,protected-access
    from multiprocessing.forkserver import _forkserver

    return _forkserver._forkserver_alive_fd is not None  # type: ignore[attr-defined]


def setup_environment(load_mode: LoadMode = "all") -> None:
    """
    Initialize the forkserver with preloaded modules and load modules in the
    main process.

    Sets the global start method to "forkserver" as the preferred default.
    Some deps (e.g. cloudfiles, taskqueue) force `spawn` at runtime; that's
    fine. Our parallel pools always use an explicit forkserver context.

    In a child process that inherited a running forkserver daemon, skip
    daemon init — the daemon is already preloaded.

    Args:
        load_mode: Which modules to load ("all", "inference", "training", "try")
    """
    if _inherited_forkserver_daemon():
        logger.info("Reusing inherited forkserver daemon; skipping init.")
        return

    current_start_method = multiprocessing.get_start_method(allow_none=True)
    if current_start_method is None:  # pragma: no cover
        multiprocessing.set_start_method("forkserver")  # type: ignore[unreachable]
    elif current_start_method == "fork":
        warnings.warn(
            "The global multiprocessing start method is set to 'fork', which is "
            "unsafe around C libraries and threads. This may have been set by an "
            "earlier import. Zetta's own parallel pools always use forkserver "
            "explicitly, but bare multiprocessing calls in dependencies will use "
            "fork.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Spawn first (single-threaded parent → safe fork+exec); the daemon's
    # preload imports overlap with load_*_modules() below.
    logger.info(f"Configuring forkserver with preload module: {_PRELOAD_MODULES[load_mode]}")
    forkserver_start = time.perf_counter()
    _spawn_forkserver_daemon(load_mode)

    if load_mode == "all":
        load_all_modules()
    elif load_mode == "inference":  # pragma: no cover
        load_inference_modules()
    elif load_mode == "try":  # pragma: no cover
        try_load_train_inference()
    else:  # training  # pragma: no cover
        load_training_modules()

    _wait_for_forkserver_ready()
    logger.info(
        f"Forkserver initialized in {time.perf_counter() - forkserver_start:.2f}s "
        f"(mode: {load_mode})"
    )
