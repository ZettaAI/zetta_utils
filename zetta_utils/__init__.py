# pylint: disable=unused-import, import-outside-toplevel, broad-exception-caught, import-error
"""Zetta AI Computational Connectomics Toolkit."""

import os
import sys
import warnings
import multiprocessing

from .log import get_logger


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


# Set global multiprocessing context
MULTIPROCESSING_CONTEXT = "spawn"

# Set start method to `spawn` if not set elsewhere.
# `fork` is unsafe after gRPC/CUDA initialization; `spawn` avoids this.
if multiprocessing.get_start_method(allow_none=True) is None:
    multiprocessing.set_start_method(MULTIPROCESSING_CONTEXT)


def get_mp_context() -> multiprocessing.context.BaseContext:
    """Get the multiprocessing context for the configured start method."""
    return multiprocessing.get_context(MULTIPROCESSING_CONTEXT)

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


def _load_core_modules():
    """Load core modules that were previously imported at package level."""
    from . import log, typing, parsing, builder, common, constants
    from . import geometry, distributions, layer, ng

    # Add builder module suppression now that it's loaded
    log.add_supress_traceback_module(builder)


def load_all_modules():
    _load_core_modules()
    load_inference_modules()
    load_training_modules()
    from . import task_management


def try_load_train_inference():  # pragma: no cover
    try:
        _load_core_modules()
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.exception(e)

    try:
        load_inference_modules()

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.exception(e)

    try:
        load_training_modules()
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.exception(e)

    try:
        from . import mazepa_addons
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.exception(e)


def load_submodules():  # pragma: no cover
    from . import internal


def load_inference_modules():
    _load_core_modules()
    from . import (
        augmentations,
        convnet,
        mazepa,
        mazepa_layer_processing,
        tensor_ops,
        tensor_typing,
        tensor_mapping,
    )
    from .layer import volumetric
    from .layer.volumetric import cloudvol
    from .message_queues import sqs

    from . import mazepa_addons
    from . import message_queues
    from . import cloud_management

    load_submodules()


def load_training_modules():
    _load_core_modules()
    from . import (
        augmentations,
        convnet,
        mazepa,
        tensor_ops,
        tensor_typing,
        training,
        tensor_mapping,
    )
    from .layer import volumetric, db_layer
    from .layer.db_layer import datastore, firestore
    from .layer.volumetric import cloudvol

    from . import mazepa_addons
    from . import message_queues
    from . import cloud_management

    load_submodules()
