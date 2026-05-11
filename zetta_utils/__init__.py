# pylint: disable=unused-import, import-outside-toplevel, broad-exception-caught, import-error
"""Zetta AI Computational Connectomics Toolkit."""
import faulthandler
import multiprocessing
import os
import sys
import time
import warnings
from typing import Literal

from .log import get_logger
from .parallel import get_mp_context  # noqa: F401

faulthandler.enable(all_threads=True)


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
LoadMode = Literal["all", "inference", "training", "try", "none", "auto"]

_PRELOAD_MODULES: dict[LoadMode, str] = {
    "all": "zetta_utils.builder.preload.all",
    "inference": "zetta_utils.builder.preload.inference",
    "training": "zetta_utils.builder.preload.training",
    "try": "zetta_utils.builder.preload.try_load",
    "none": "zetta_utils.builder.preload.none",
    # "auto" is special: preload list is computed from the spec, not a
    # fixed module path. Handled in setup_environment().
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


def _spawn_forkserver_with_modules(modules: list[str]) -> None:  # pragma: no cover
    """Spawn the daemon with an explicit preload list (not a fixed module path).

    Only called from the auto-mode branch of setup_environment (itself
    pragma'd because exercising it requires the parent's autouse conftest
    fixture to NOT have already set up 'all', which conflicts with our
    test infrastructure). End-to-end auto-mode coverage lives in the
    subprocess test in tests/unit/builder/test_auto_mode.py.
    """
    # pylint: disable=import-outside-toplevel,protected-access
    from multiprocessing.forkserver import _forkserver

    multiprocessing.set_forkserver_preload(modules)
    _forkserver.ensure_running()


def _setup_auto(
    cue_path: str | None,
    forkserver_start: float,
) -> bool:  # pragma: no cover
    """Auto mode: compute preload set + spawn daemon. Returns False if it must
    fall back to 'all' (no cue_path supplied)."""
    # pylint: disable=import-outside-toplevel
    if cue_path is None:
        warnings.warn(
            "setup_environment(load_mode='auto') requires cue_path; "
            "falling back to 'all'.",
            RuntimeWarning,
            stacklevel=2,
        )
        return False

    from zetta_utils.builder.preload import compute_preload_set
    from zetta_utils.parsing import cue as _cue
    from zetta_utils.parsing.spec_scan import extract_types

    spec = _cue.load(cue_path)
    scan_result = extract_types(spec)
    if scan_result.has_dynamic_types:
        logger.warning(
            "auto preload: spec contains dynamic @type values; "
            "lookup-miss fallback will handle them at build time."
        )
    preload_modules = compute_preload_set(scan_result.names())
    logger.info(
        f"auto preload: {len(preload_modules)} module(s) "
        f"({len(scan_result.types)} @type refs in spec)"
    )

    _spawn_forkserver_with_modules(preload_modules)
    _wait_for_forkserver_ready()
    logger.info(
        f"Forkserver initialized in "
        f"{time.perf_counter() - forkserver_start:.2f}s (mode: auto)"
    )
    return True


def setup_environment(
    load_mode: LoadMode = "all",
    cue_path: str | None = None,
) -> None:
    """
    Initialize the forkserver with preloaded modules and load modules in the
    main process.

    Sets the global start method to "forkserver" as the preferred default.
    Some deps (e.g. cloudfiles, taskqueue) force `spawn` at runtime; that's
    fine. Our parallel pools always use an explicit forkserver context.

    In a child process that inherited a running forkserver daemon, skip
    daemon init — the daemon is already preloaded.

    Args:
        load_mode: Which modules to load. "all"/"inference"/"training"/"try"
            preload fixed module bundles. "none" registers the always-eager
            set and lazy-spawns the daemon on first fork. "auto" requires
            `cue_path`; scans the CUE for @type literals and preloads exactly
            the modules needed via the static registry index.
        cue_path: Required when load_mode == "auto". Path to the CUE file
            being executed. Ignored otherwise.
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

    forkserver_start = time.perf_counter()

    if load_mode == "auto":  # pragma: no cover
        if _setup_auto(cue_path, forkserver_start):
            return
        load_mode = "all"

    # Spawn first (single-threaded parent → safe fork+exec); the daemon's
    # preload imports overlap with load_*_modules() below.
    logger.info(f"Configuring forkserver with preload module: {_PRELOAD_MODULES[load_mode]}")
    _spawn_forkserver_daemon(load_mode)

    if load_mode == "all":
        load_all_modules()
    elif load_mode == "inference":  # pragma: no cover
        load_inference_modules()
    elif load_mode == "try":  # pragma: no cover
        try_load_train_inference()
    elif load_mode == "none":  # pragma: no cover
        # Skip eager loading; registry.get_matching_entry will lazy-import
        # modules on demand via the static index. Daemon still spawns
        # eagerly so worker latency is paid up front, not at first fork.
        pass
    else:  # training  # pragma: no cover
        load_training_modules()

    _wait_for_forkserver_ready()
    logger.info(
        f"Forkserver initialized in {time.perf_counter() - forkserver_start:.2f}s "
        f"(mode: {load_mode})"
    )
